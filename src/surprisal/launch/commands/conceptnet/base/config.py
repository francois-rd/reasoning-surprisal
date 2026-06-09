from dataclasses import dataclass, field
import os

from .....core import AggregatorOption, SystemPromptID, UserTemplateID
from .....llms import Nickname
from .....parsing import ParserID

from .core import ConceptNetTerm, LinguisticsID, RelationType, Query
from .rankers import RankerID

VariantID = str
_FactualVariantID = "factual"


@dataclass
class VariantInfo:
    ranker_id: RankerID = "noop"  # Which ranker to use.
    linguistics_id: LinguisticsID = "no_ling"  # Which features to use.
    system_prompt_id: SystemPromptID = "cnet_simple"  # Which system prompt to use.
    user_template_id: UserTemplateID = "cnet_simple"  # Which user template to use.
    parser_id: ParserID = "true_false"  # Which parser to use.


@dataclass
class VariantsConfig:
    variants: dict[VariantID, VariantInfo] = field(
        default_factory=lambda: {_FactualVariantID: VariantInfo()}
    )


@dataclass
class TripletVariantCluster:
    factual_query: Query
    factual_target: ConceptNetTerm
    non_factual_candidates: dict[VariantID, list[ConceptNetTerm]] = field(
        default_factory=dict
    )


@dataclass
class Config:
    # General fields.
    verbose: bool = True  # Whether to be verbose during processing steps.
    llms: list[Nickname] = field(default_factory=list)  # Which LLMs to use.

    # Preprocessing fields.
    factual_variant_id: VariantID = _FactualVariantID  # The baseline variant to use.
    subsampling_per_relation: int = 200  # Max number of samples per relation type.
    preprocess_nf_threshold: int = 5  # Min number of NF candidates to keep a pairing.
    preprocess_seed: int = 42  # Random seed for reproducibility.
    relation_type: RelationType = "IsA"  # Which relation type to focus on. This
    # is needed only during preprocessing to avoid OOM issues.

    # Inference fields.
    inference_variant_id: VariantID = _FactualVariantID  # Which variant to run.
    prompt_batch_size: int = 10  # How many samples to run at once.
    checkpoint_frequency: float = 5 * 60  # E.g., save every 5 minutes.
    chosen_only_logprob: bool = True  # Keep all logprobs or teacher-forced only.

    # Postprocessing fields.
    collate_variants: list[VariantID] = field(  # Which inference variants to collate.
        default_factory=lambda: [_FactualVariantID]
    )

    # Analysis fields.
    aggregators: list[AggregatorOption] = field(  # Which logprob aggs to use.
        default_factory=lambda: [AggregatorOption.SUM, AggregatorOption.MIN]
    )
    flip_logprobs: bool = True  # Whether to flip logprob sign from - to +.

    def _build_id(self, variant_id: VariantID | None = None):
        prefix = [variant_id] if variant_id is not None else []
        return "-".join(
            prefix
            + [
                str(self.subsampling_per_relation),
                str(self.preprocess_seed),
                str(self.preprocess_nf_threshold),
            ]
        )

    def preprocess_dir(self, root: str) -> str:
        return str(os.path.join(root, "preprocess", self._build_id()))

    def prompts_file(self, root: str, v_id: VariantID) -> str:
        return str(os.path.join(root, "prompts", self._build_id(v_id))) + ".jsonl"

    def llm_output_dir(self, root: str, nickname: Nickname) -> str:
        return str(os.path.join(root, "output", nickname, self._build_id()))

    def postprocess_dir(self, root: str) -> str:
        return str(os.path.join(root, "postprocess", self._build_id()))

    def analysis_dir(self, root: str) -> str:
        return str(os.path.join(root, "analysis", self._build_id()))

    def plots_dir(self, root: str) -> str:
        return str(os.path.join(root, "plots", self._build_id()))
