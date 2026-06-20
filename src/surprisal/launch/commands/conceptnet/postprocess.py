from datetime import datetime
from enum import Enum
import os

from coma import command
import pandas as pd

from ....llms import Inference, Nickname, flatten
from ....core import AggregatorOption, Logprobs, UserTemplatesConfig
from ....io import (
    ConditionalPrinter,
    PathConfig,
    ensure_path,
    init_logger,
    load_dataclass_jsonl,
    walk_files,
)

from .base import (
    ConceptNetFormatter,
    Config,
    TermFormatter,
    Triplet,
    VariantID,
    VariantsConfig,
)
from .infer import AdditionalDataFields


class DFCols(Enum):
    SURPRISAL = "Surprisal"
    VARIANT_ID = "VariantID"
    RELATION_TYPE = "RelationType"
    GROUP_ID = "GroupID"
    P_VALUE = "PValue"
    LLM = "LLM"


class DataLoader:
    def __init__(
        self,
        formatter: ConceptNetFormatter,
        log_dir: str,
        flip_logprobs: bool,
        aggregator: AggregatorOption,
    ):
        self.formatter = formatter
        self.log_dir = log_dir
        self.flip = flip_logprobs
        self.agg = aggregator
        self.data = self.logger = self.llm = None

    def load(self, inference_path: str, llm: Nickname) -> pd.DataFrame:
        self.data, self.logger, self.llm = {}, None, llm
        for inference in load_dataclass_jsonl(inference_path, t=Inference):
            if inference.error_message is not None:
                continue
            self._fill(inference)
        if not self.data:
            raise ValueError(f"Empty data frame for: {inference_path}.")
        return pd.DataFrame(self.data)

    def _get_logprobs(self, inference: Inference) -> list[float] | None:
        logprobs = Logprobs.from_dict(
            inference.derived_data["prompt_logprobs"], self.flip
        )
        triplet_key = AdditionalDataFields.TRIPLET.value
        triplet = Triplet(**inference.prompt_data.additional_data[triplet_key])
        target = self.formatter.formatter.ensure_plain_text(triplet.target)
        seqs = list(logprobs.indices_of(target))
        desired_seqs = [s for s in seqs if self.formatter.is_desired_target(s)]
        if len(seqs) == 1:
            return seqs[0].to_chosen_logprobs()
        elif len(desired_seqs) == 1:
            return desired_seqs[0].to_chosen_logprobs()
        else:
            logger_name = f"cnet.postprocess.{self.llm}"
            now = datetime.now().isoformat()
            log_file = os.path.join(self.log_dir, f"{self.llm}-{now}.log")
            self.logger = self.logger or init_logger(logger_name, log_file)
            no_line_breaks = logprobs.to_text().replace("\n", "<NL>")
            self.logger.info(
                f"GroupID={inference.prompt_data.group_id}: Target '{target}' "
                f"has {len(seqs)} occurrences in '{no_line_breaks}'."
            )
            return None

    def _fill(self, inference: Inference) -> None:
        logprobs = self._get_logprobs(inference)
        if logprobs is None:
            # Don't add if logprobs could not be found.
            return

        v_id_key = AdditionalDataFields.VARIANT_ID.value
        triplet_key = AdditionalDataFields.TRIPLET.value
        variant_id = inference.prompt_data.additional_data[v_id_key]
        triplet = Triplet(**inference.prompt_data.additional_data[triplet_key])

        self._do_fill(DFCols.SURPRISAL, self.agg.aggregate(logprobs))
        self._do_fill(DFCols.VARIANT_ID, variant_id)
        self._do_fill(DFCols.RELATION_TYPE, triplet.relation)
        self._do_fill(DFCols.GROUP_ID, inference.prompt_data.group_id)

    def _do_fill(self, col: DFCols, value) -> None:
        self.data.setdefault(col.value, []).append(value)


@command(name="cnet.postprocess")
class ConceptNetPostProcess:
    def __init__(
        self,
        path: PathConfig,
        conceptnet: Config,
        cnet_variants: VariantsConfig,
        user_templates: UserTemplatesConfig,
    ):
        self.path = path
        self.cfg = conceptnet
        self.variants = cnet_variants.variants
        self.templates = user_templates.templates
        self.print = ConditionalPrinter(self.cfg.verbose)

    def _skip(self, test_variant_id: VariantID) -> bool:
        for variant_id in self.cfg.collate_variants:
            if test_variant_id == variant_id:
                return False
        return True

    def run(self):
        for nickname in self.cfg.llms:
            nickname = flatten(nickname)
            all_data = []
            self.print(f"Postprocessing model={nickname}...")
            llm_output_dir = self.cfg.llm_output_dir(self.path.cnet_exp_dir, nickname)
            for walk in walk_files(llm_output_dir):
                inference_path, variant_id = walk.path, walk.no_ext()
                if self._skip(variant_id):
                    continue
                self.print(f"    Loading data for variant={variant_id}...")
                df = self._create_data_loader(variant_id).load(inference_path, nickname)
                all_data.append(df)
                self.print("    Done.")
            self.print("    Collating all loaded data...")
            out_dir = self.cfg.postprocess_dir(self.path.cnet_exp_dir)
            out_file = ensure_path(os.path.join(out_dir, nickname + ".csv"))
            pd.concat(all_data, ignore_index=True).to_csv(out_file, index=False)
            self.print("    Done.")

    def _create_data_loader(self, variant_id: VariantID) -> DataLoader:
        template = self.templates[self.variants[variant_id].user_template_id].template
        return DataLoader(
            formatter=ConceptNetFormatter(
                template=template,
                formatter=TermFormatter(language="en"),
            ),
            log_dir=self.cfg.analysis_dir(self.path.cnet_exp_dir),
            flip_logprobs=self.cfg.flip_logprobs,
            aggregator=self.cfg.aggregator,
        )
