from enum import Enum
import os

from coma import InvocationData, command, preload

from ....core import SystemPromptsConfig, UserTemplatesConfig
from ....io import (
    ConditionalPrinter,
    PathConfig,
    load_dataclass_jsonl,
    save_dataclass_jsonl,
    walk_files,
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
from ....parsing import ParserManager, ParsersConfig

from .base import (
    Config,
    ConceptNetFormatter,
    ConceptNetTerm,
    Ranker,
    RankerManager,
    RankersConfig,
    TermFormatter,
    Triplet,
    TripletVariantCluster,
    VariantID,
    VariantsConfig,
)


class AdditionalDataFields(Enum):
    TRIPLET = "triplet"
    VARIANT_ID = "variant_id"


@command(name="cnet.make.prompts")
class ConceptNetMakePrompts:
    def __init__(
        self,
        path: PathConfig,
        conceptnet: Config,
        cnet_variants: VariantsConfig,
        rankers: RankersConfig,
    ):
        self.path = path
        self.cfg = conceptnet
        self.variants = cnet_variants.variants
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.id_generator = IDGenerator()
        self.formatter = TermFormatter(language="en")
        self.rankers = self._create_rankers(RankerManager(rankers))

    def _create_rankers(self, manager: RankerManager) -> dict[VariantID, Ranker]:
        return {v_id: manager.get(v.ranker_id) for v_id, v in self.variants.items()}

    def run(self):
        data = self._load_data()
        self.print("Generating prompts...")
        for variant_id, prompts in self._make_prompts(data).items():
            file_name = self.cfg.prompts_file(self.path.cnet_exp_dir, variant_id)
            save_dataclass_jsonl(file_name, *prompts, ensure_ascii=False)
        self.print("Done.")

    def _load_data(self) -> list[TripletVariantCluster]:
        data = []
        self.print("Loading preprocessed data...")
        for walk in walk_files(self.cfg.preprocess_dir(self.path.cnet_exp_dir)):
            self.print(f"    Loading data for {walk.base}...")
            for cluster in load_dataclass_jsonl(walk.path, t=TripletVariantCluster):
                if self._prune_data(cluster):
                    data.append(cluster)
            self.print("    Done.")
        self.print("Done.")
        return data

    def _prune_data(self, cluster: TripletVariantCluster) -> bool:
        for variant_id in cluster.non_factual_candidates.keys():
            top_ranked_terms = self.rankers[variant_id](
                result=set(cluster.non_factual_candidates[variant_id]),
                formatter=self.formatter,
                factual_target=cluster.factual_target,
            )
            if len(top_ranked_terms) == 0:
                return False
            top_ranked_term, metric = top_ranked_terms[0]
            if self.rankers[variant_id].is_worst_outcome(metric):
                # Skipping because no candidate is good.
                return False
            cluster.non_factual_candidates[variant_id] = [top_ranked_term]
        return True

    def _make_prompts(
        self, clusters: list[TripletVariantCluster]
    ) -> dict[VariantID, list[PromptData]]:
        prompts = {}
        for cluster in clusters:
            self.id_generator.next_group_id()  # Reset for each new group.
            variants_and_terms = {self.cfg.factual_variant_id: [cluster.factual_target]}
            variants_and_terms.update(cluster.non_factual_candidates.items())
            for variant_id, term in variants_and_terms.items():
                prompts.setdefault(variant_id, []).append(
                    self._do_make_prompt(cluster, variant_id, term[0])
                )
        return prompts

    def _do_make_prompt(
        self,
        cluster: TripletVariantCluster,
        variant_id: VariantID,
        top_ranked_term: ConceptNetTerm,
    ) -> PromptData:
        return PromptData(
            messages=[
                Message(MessageType.SYSTEM, "PLACEHOLDER"),
                Message(MessageType.USER, "PLACEHOLDER"),
            ],
            prompt_id=self.id_generator.next_prompt_id(),
            group_id=self.id_generator.next_group_id(no_increment=True),
            additional_data={
                AdditionalDataFields.TRIPLET.value: Triplet(
                    source=cluster.factual_query.source_term,
                    relation=cluster.factual_query.relation_type,
                    target=top_ranked_term,
                ),
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
    name="cnet.infer",
    pre_config_hook=pre_config_hook,
    dummy=DummyConfig,
    openai=OpenAIConfig,
)
class ConceptNetInfer:
    def __init__(
        self,
        path: PathConfig,
        conceptnet: Config,
        cnet_variants: VariantsConfig,
        parsers: ParsersConfig,
        llms: LLMsConfig,
        system_prompts: SystemPromptsConfig,
        user_templates: UserTemplatesConfig,
        llm_cfg_placeholder,
    ):
        self.path = path
        self.cfg = conceptnet
        self.llms = llms
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.variant_id = self.cfg.inference_variant_id
        variant = cnet_variants.variants[self.variant_id]
        self.system_prompt = system_prompts.prompts[variant.system_prompt_id]
        self.template = user_templates.templates[variant.user_template_id]
        out = self.cfg.llm_output_dir(self.path.cnet_exp_dir, flatten(self.llms.llm))
        self.out_file = os.path.join(out, f"{self.variant_id}.jsonl")
        self.formatter = ConceptNetFormatter(
            template=self.template.template,
            formatter=TermFormatter(language="en"),
        )
        self.infer = CheckpointedParallelInference(
            infer=ParallelInference(
                parser=ParserManager(parsers).get(variant.parser_id),
                llms=llms,
                llm_cfg=llm_cfg_placeholder,  # Passed to *all* LLMs' init.
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
        prompts_file = self.cfg.prompts_file(self.path.cnet_exp_dir, self.variant_id)
        for prompt_data in load_dataclass_jsonl(prompts_file, t=PromptData):
            v_id_key = AdditionalDataFields.VARIANT_ID.value
            triplet_key = AdditionalDataFields.TRIPLET.value
            if self.variant_id != prompt_data.additional_data[v_id_key]:
                continue
            if self.infer.skip(prompt_data):
                continue
            triplet = Triplet(**prompt_data.additional_data[triplet_key])
            prompt_data.messages[0].text = self.system_prompt
            prompt_data.messages[1].text = self.formatter(triplet)
            prompts.append(prompt_data)
        self.print("Done.")
        return prompts

    def run(self):
        self.infer(self.prompts, add_prompt_logprobs=True)
