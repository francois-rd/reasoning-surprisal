from enum import Enum
import os

from coma import InvocationData, command, preload

from ....core import SystemPromptsConfig, UserTemplatesConfig
from ....io import (
    ConditionalPrinter,
    PathConfig,
    load_dataclass_jsonl,
    save_dataclass_jsonl,
)
from ....llms import (
    CheckpointedParallelInference,
    DummyConfig,
    IDGenerator,
    LLMsConfig,
    LLMImplementation,
    Message,
    MessageType,
    OpenAIConfig,
    ParallelInference,
    PromptData,
    flatten,
)
from ....parsing import NoOutputParser

from .base import (
    AccordLoader,
    Config,
    Instance,
    VariantID,
    VariantInfo,
    VariantsConfig,
)


class AdditionalDataFields(Enum):
    ACCORD_ID = "accord_id"
    CSQA_LABEL = "csqa_label"
    VARIANT_ID = "variant_id"


@command(name="accord.make.prompts")
class AccordMakePrompts:
    def __init__(
        self,
        path: PathConfig,
        accord: Config,
        accord_variants: VariantsConfig,
        user_templates: UserTemplatesConfig,
    ):
        self.path = path
        self.cfg = accord
        self.variants = accord_variants.variants
        self.user_templates = user_templates
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.id_generator = IDGenerator()

    def run(self):
        for variant_id, variant in self.variants.items():
            self.print("Generating prompts for:", variant_id)
            prompts = self._make_prompts(variant_id, variant)
            file_name = self.cfg.prompts_file(self.path.accord_exp_dir, variant_id)
            save_dataclass_jsonl(file_name, *prompts, ensure_ascii=False)
        self.print("Done.")

    def _make_prompts(
        self, variant_id: VariantID, variant: VariantInfo
    ) -> list[PromptData]:
        instance_groups, prompts = {}, []

        # Sort instances into Factual and Non-Factual pairs. Baseline doesn't have
        # non-factual pairs, but can still use the same logic.
        for instance in AccordLoader(variant, self.path, self.user_templates).load():
            group_id = instance.get_factuality_independent_group_id()
            instance_groups.setdefault(group_id, []).append(instance)

        # Remap the grouping information as prompts are constructed from instances.
        for group in instance_groups.values():
            self.id_generator.next_group_id()  # Reset for each new group.
            prompts.extend([self._do_make_prompt(i, variant_id) for i in group])
        return prompts

    def _do_make_prompt(self, instance: Instance, variant_id: VariantID) -> PromptData:
        return PromptData(
            messages=[
                Message(MessageType.SYSTEM, "PLACEHOLDER"),
                Message(MessageType.USER, "PLACEHOLDER"),
            ],
            prompt_id=self.id_generator.next_prompt_id(),
            group_id=self.id_generator.next_group_id(no_increment=True),
            additional_data={
                # This uniquely maps each Prompt to a unique Instance, which stores
                # all possible relevant information. That is, we don't need to store
                # the entire MetaData here to have access to it later. Only this ID.
                AdditionalDataFields.ACCORD_ID.value: instance.meta_data.id,
                # The only thing not in MetaData in the CSQA Label.
                AdditionalDataFields.CSQA_LABEL.value: instance.csqa_label,
                # Similarly, this records all variant information for later access.
                AdditionalDataFields.VARIANT_ID.value: variant_id,
            },
        )


def pre_config_hook(data: InvocationData) -> None:
    preload(data, "llms")
    llms: LLMsConfig = data.parameters.get_config("llms").get_latest()
    if llms.implementation == LLMImplementation.MISSING:
        raise ValueError("Missing LLM implementation.")
    elif llms.implementation == LLMImplementation.DUMMY:
        llm_cfg_name = "dummy"
        drop_cfg_names = ["openai"]
    elif llms.implementation == LLMImplementation.OPENAI:
        llm_cfg_name = "openai"
        drop_cfg_names = ["dummy"]
    else:
        raise ValueError(f"Unsupported implementation: {llms.implementation}")
    preload(data, llm_cfg_name)
    config = data.parameters.get_config(llm_cfg_name)
    llm_cfg = config.as_primitive(config.get_latest_key())
    data.parameters.replace("llm_cfg_placeholder", llm_cfg)
    data.parameters.delete(*drop_cfg_names)


@command(
    name="accord.infer",
    pre_config_hook=pre_config_hook,
    dummy=DummyConfig,
    openai=OpenAIConfig,
)
class AccordInfer:
    def __init__(
        self,
        path: PathConfig,
        accord: Config,
        accord_variants: VariantsConfig,
        llms: LLMsConfig,
        system_prompts: SystemPromptsConfig,
        user_templates: UserTemplatesConfig,
        llm_cfg_placeholder,
    ):
        self.path = path
        self.cfg = accord
        self.llms = llms
        self.user_templates = user_templates
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.variant_id = self.cfg.inference_variant_id
        self.variant = accord_variants.get_variant(self.variant_id)
        self.system_prompt = self.variant.get_system_prompt(system_prompts)
        self.template = self.variant.get_user_template(user_templates)
        out = self.cfg.llm_output_dir(self.path.accord_exp_dir, flatten(self.llms.llm))
        self.out_file = os.path.join(out, f"{self.variant_id}.jsonl")
        self.infer = CheckpointedParallelInference(
            infer=ParallelInference(
                parser=NoOutputParser(),
                llms=llms,
                llm_cfg=llm_cfg_placeholder,  # Passed to all LLMs' init.
                out_dir=out,  # Passed to OpenaiLLM's init.
                chosen_only=self.cfg.chosen_only_logprob,  # Passed to OpenaiLLM's init.
                trim_indicator=self.template.indicator,  # Passed to OpenaiLLM's init.
            ),
            out_file=self.out_file,
            batch_size=self.cfg.prompt_batch_size,
            verbose=self.cfg.verbose,
            frequency=self.cfg.checkpoint_frequency,
        )
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> list[PromptData]:
        # Load prompts, skip checkpoints, and replace prompt placeholders.
        self.print("Loading data...")
        prompts = []
        instances = AccordLoader(self.variant, self.path, self.user_templates).load()
        instances = {instance.meta_data.id: instance.text for instance in instances}
        prompts_file = self.cfg.prompts_file(self.path.accord_exp_dir, self.variant_id)
        for prompt_data in load_dataclass_jsonl(prompts_file, t=PromptData):
            accord_id_key = AdditionalDataFields.ACCORD_ID.value
            variant_id_key = AdditionalDataFields.VARIANT_ID.value
            if self.variant_id != prompt_data.additional_data[variant_id_key]:
                continue
            if self.infer.skip(prompt_data):
                continue
            text = instances[prompt_data.additional_data[accord_id_key]]
            prompt_data.messages[0].text = self.system_prompt
            prompt_data.messages[1].text = text
            prompts.append(prompt_data)
        self.print("Done.")
        return prompts

    def run(self):
        self.infer(self.prompts, add_prompt_logprobs=True)
