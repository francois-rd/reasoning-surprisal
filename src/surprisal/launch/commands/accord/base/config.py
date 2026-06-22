from dataclasses import dataclass, field
from enum import Enum
import os

from coma import command

from .....core import (
    AggregatorOption,
    SystemPrompt,
    SystemPromptID,
    SystemPromptsConfig,
    UserTemplate,
    UserTemplateID,
    UserTemplatesConfig,
)
from .....io import PathConfig, dumps_dataclasses, load_dataclass_jsonl
from .....llms import Nickname

from .dataclasses import (
    CaseLink,
    Instance,
    MetaData,
    SurfaceForms,
    CsqaBase,
)
from .surfacer import (
    InstanceSurfacer,
    OrderingSurfacer,
    QADataSurfacer,
    StatementSurfacer,
    TermSurfacer,
    TextSurfacer,
)


class AccordSubset(Enum):
    BASELINE = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5

    def get_meta_data_file(self, path: PathConfig) -> str:
        if self == AccordSubset.BASELINE:
            return path.accord_baseline_file
        elif self == AccordSubset.ONE:
            return path.accord_tree_size_1_file
        elif self == AccordSubset.TWO:
            return path.accord_tree_size_2_file
        elif self == AccordSubset.THREE:
            return path.accord_tree_size_3_file
        elif self == AccordSubset.FOUR:
            return path.accord_tree_size_4_file
        elif self == AccordSubset.FIVE:
            return path.accord_tree_size_5_file
        else:
            raise ValueError(f"Unsupported ACCORD subset: {self}")

    def get_reductions_file(self, path: PathConfig) -> str | None:
        return None if self == AccordSubset.BASELINE else path.accord_reductions_file


VariantID = str
_BaselineVariantID = "baseline"


@dataclass
class VariantInfo:
    subset: AccordSubset = AccordSubset.BASELINE  # Which subset to use.
    system_prompt_id: SystemPromptID = "accord_baseline"  # Which system prompt to use.
    user_template_id: UserTemplateID = "accord_baseline"  # Which user template to use.

    def get_system_prompt(self, system_prompts: SystemPromptsConfig) -> SystemPrompt:
        return system_prompts.prompts[self.system_prompt_id]

    def get_user_template(self, user_templates: UserTemplatesConfig) -> UserTemplate:
        return user_templates.templates[self.user_template_id]


@dataclass
class VariantsConfig:
    variants: dict[VariantID, VariantInfo] = field(
        default_factory=lambda: {_BaselineVariantID: VariantInfo()}
    )

    def get_variant(self, variant_id: VariantID) -> VariantInfo:
        return self.variants[variant_id]


@dataclass
class Config:
    # General fields.
    verbose: bool = True  # Whether to be verbose during processing steps.
    seed: int = 42  # Random seed for reproducibility.
    llms: list[Nickname] = field(default_factory=list)  # Which LLMs to use.

    # Inference fields.
    inference_variant_id: VariantID = _BaselineVariantID  # The variant to use.
    prompt_batch_size: int = 10  # How many samples to run at once.
    checkpoint_frequency: float = 5 * 60  # E.g., save every 5 minutes.
    chosen_only_logprob: bool = True  # Keep all logprobs or teacher-forced only.

    # Postprocessing and analysis fields.
    collate_variants: list[VariantID] = field(  # Which inference variants to collate.
        default_factory=lambda: [_BaselineVariantID]
    )
    aggregator: AggregatorOption = AggregatorOption.SUM  # Which logprob agg to use.
    flip_logprobs: bool = True  # Whether to flip logprob sign from - to +.
    show_assumption_plots: bool = False  # Whether to show homoscedasticity and
    # normality plots for linear mixed models.
    create_violin_plots: bool = False  # Whether to create detailed violin plots for
    # each LLM. Summary bar charts are created no matter what.

    @staticmethod
    def prompts_file(root: str, variant_id: VariantID) -> str:
        return str(os.path.join(root, "prompts", variant_id)) + ".jsonl"

    @staticmethod
    def llm_output_dir(root: str, nickname: Nickname) -> str:
        return str(os.path.join(root, "output", nickname))

    @staticmethod
    def postprocess_dir(root: str) -> str:
        return str(os.path.join(root, "postprocess"))

    @staticmethod
    def analysis_dir(root: str) -> str:
        return str(os.path.join(root, "analysis"))

    @staticmethod
    def plots_dir(root: str) -> str:
        return str(os.path.join(root, "plots"))


class AccordLoader:
    """
    Loads and then preprocesses ACCORD_CSQA MetaData into ACCORD_CSQA Instances
    matching those used to generate and evaluate ACCORD_CSQA from the ACCORD paper.
    """

    def __init__(
        self,
        variant: VariantInfo,
        path: PathConfig,
        user_templates: UserTemplatesConfig,
    ):
        self.data_file = variant.subset.get_meta_data_file(path)
        self.forms_file = variant.subset.get_reductions_file(path)
        self.csqa_file = path.accord_csqa_file
        self.instruction_prompt = variant.get_user_template(user_templates).template
        self.surfacer: InstanceSurfacer | None = None
        self.variant = variant

    def get_subset(self) -> AccordSubset:
        return self.variant.subset

    def load(self) -> list[Instance]:
        self.surfacer = self._create_surfacer()
        meta_datas = load_dataclass_jsonl(self.data_file, t=MetaData)
        csqa = load_dataclass_jsonl(self.csqa_file, t=CsqaBase)
        csqa = {instance.identifier: instance for instance in csqa}
        return [
            Instance(self.surfacer(md), csqa[md.qa_id].correct_answer_label, md)
            for md in meta_datas
        ]

    def _load_surface_forms(self) -> SurfaceForms:
        """
        Loads and registers CaseLink subsumed surface forms from a CSV-formatted file.

        Empty lines are skipped, as are lines starting with the comment string.

        Relevant headers from the CSV file are:
            "relation1_or_pairing" -> CaseLink.r1_type
            "relation2" -> CaseLink.r2_type
            "case" -> CaseLink.case
            "relation2_surface_form" -> subsumed surface form for associated CaseLink
        """
        line_comment = "#"
        with open(self.forms_file, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            lines = [
                line
                for line in f.readlines()
                if line.strip() and not line.strip().startswith(line_comment)
            ]
        forms = SurfaceForms()
        for r in [dict(zip(header, line.strip().split(","))) for line in lines]:
            cl = CaseLink(r["relation1_or_pairing"], r["relation2"], int(r["case"]))
            forms.register(cl, r["relation2_surface_form"])
        return forms

    def _create_surfacer(self) -> InstanceSurfacer:
        """
        Creates an InstanceSurfacer with all config parameters matching those used
        to generate and evaluate ACCORD_CSQA from the ACCORD paper.
        """
        forms = None if self.forms_file is None else self._load_surface_forms()
        ordering_surfacer = None
        if forms is not None:
            ordering_surfacer = OrderingSurfacer(
                prefix="Statements:\n",
                statement_separator="\n",
                statement_surfacer=StatementSurfacer(
                    prefix="- ",
                    term_surfacer=TermSurfacer(prefix="[", suffix="]"),
                    forms=forms,
                ),
            )
        return InstanceSurfacer(
            prefix="",
            surfacer_separator="\n",
            prefix_surfacer=TextSurfacer(
                prefix="Instructions:\n",
                text=self.instruction_prompt,
            ),
            ordering_surfacer=ordering_surfacer,
            qa_data_surfacer=QADataSurfacer(
                prefix="Question:\n",
                question_answer_separator="\n",
                answer_choice_separator="    ",
                answer_choice_formatter="{}: {}",
            ),
            suffix_surfacer=TextSurfacer(prefix="Answer:\n", text=""),
        )


@command(name="test.accord.loader")
def cmd(
    path: PathConfig,
    accord_variants: VariantsConfig,
    user_templates: UserTemplatesConfig,
):
    data = []
    for variant_id, variant in accord_variants.variants.items():
        data.append(AccordLoader(variant, path, user_templates).load()[1])
    print(dumps_dataclasses(*data, indent=4))
