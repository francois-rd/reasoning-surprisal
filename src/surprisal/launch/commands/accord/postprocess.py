from typing import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import os

from coma import command
import pandas as pd

from ....core import AggregatorOption, Logprobs, SpacedSubsequence, UserTemplatesConfig
from ....llms import Inference, Nickname, flatten
from ....io import (
    ConditionalPrinter,
    PathConfig,
    ensure_path,
    init_logger,
    load_dataclass_jsonl,
    walk_files,
)

from .base import (
    AccordID,
    AccordLoader,
    AccordSubset,
    Config,
    CsqaID,
    Label,
    MetaData,
    Term,
    VariantID,
    VariantsConfig,
)
from .infer import AdditionalDataFields


class DFCols(Enum):
    # Identifiers.
    VARIANT_ID = "VariantID"
    GROUP_ID = "GroupID"
    ACCORD_ID = "AccordID"
    CSQA_ID = "CsqaID"
    LLM = "LLM"

    # Factors.
    IS_FACTUAL = "IsFactual"
    REASONING_HOPS = "ReasoningHops"
    DISTRACTORS = "Distractors"
    SUBSET = "Subset"
    ACCORD_LABEL = "AccordLabel"
    CSQA_LABEL = "CsqaLabel"

    # Label Metrics.
    SURPRISAL_A = "SurprisalA"
    SURPRISAL_B = "SurprisalB"
    SURPRISAL_C = "SurprisalC"
    SURPRISAL_D = "SurprisalD"
    SURPRISAL_E = "SurprisalE"

    # Final Metrics.
    SURPRISAL = "Surprisal"
    P_VALUE = "PValue"


@dataclass
class _CurrentData:
    llm: Nickname
    inference: Inference
    meta_data: MetaData
    subset: AccordSubset
    aggregator: AggregatorOption
    logprobs: Logprobs
    question_end_index: int | None = None
    answer_tag_start_index: int | None = None
    logprob_of_answer_choices: dict[Label, list[float]] | None = None

    def get_question(self) -> str:
        return self.meta_data.question

    def get_answer_choices(self) -> dict[Label, Term]:
        return self.meta_data.answer_choices

    def get_csqa_label(self) -> Label:
        csqa_key = AdditionalDataFields.CSQA_LABEL.value
        return self.inference.prompt_data.additional_data[csqa_key]

    def get_accord_label(self) -> Label:
        return self.meta_data.label

    def is_factual(self) -> bool:
        return self.meta_data.is_factual(self.get_csqa_label())

    def is_baseline(self) -> bool:
        return self.meta_data.is_baseline()

    def get_reasoning_hops(self) -> int:
        reduction_cases = self.meta_data.reduction_cases
        return 0 if reduction_cases is None else len(reduction_cases) + 1

    def get_distractors(self) -> int | None:
        if self.is_baseline():
            return None
        return self.subset.value - self.get_reasoning_hops()

    def get_variant_id(self) -> VariantID:
        v_id_key = AdditionalDataFields.VARIANT_ID.value
        return self.inference.prompt_data.additional_data[v_id_key]

    def get_accord_id(self) -> AccordID:
        accord_id_key = AdditionalDataFields.ACCORD_ID.value
        return self.inference.prompt_data.additional_data[accord_id_key]

    def get_group_id(self) -> int:
        return self.inference.prompt_data.group_id

    def get_csqa_id(self) -> CsqaID:
        return self.meta_data.qa_id


class VariantLoader:
    def __init__(
        self,
        variant_id: VariantID,
        accord_loader: AccordLoader,
        logging_dir: str,
        flip_logprobs: bool,
        aggregator: AggregatorOption,
    ):
        self.variant_id = variant_id
        self.accord_loader = accord_loader
        self.logging_dir = logging_dir
        self.flip = flip_logprobs
        self.agg = aggregator
        self.logger = None
        self.meta_datas = {i.meta_data.id: i.meta_data for i in accord_loader.load()}

    def load(self, inference_path: str, llm: Nickname) -> pd.DataFrame:
        data, self.logger = {}, None
        for inference in load_dataclass_jsonl(inference_path, t=Inference):
            if inference.error_message is not None:
                continue
            prompt_data = inference.prompt_data
            prompt_logprobs = inference.derived_data["prompt_logprobs"]
            a_id_key = AdditionalDataFields.ACCORD_ID.value
            meta_data = self.meta_datas[prompt_data.additional_data[a_id_key]]
            current_data = _CurrentData(
                llm=llm,
                inference=inference,
                meta_data=meta_data,
                subset=self.accord_loader.get_subset(),
                aggregator=self.agg,
                logprobs=Logprobs.from_dict(prompt_logprobs, self.flip),
            )
            success = (
                self._find_question_end_index(current_data)
                and self._find_answer_tag_start_index(current_data)
                and self._fill_logprob_of_answer_choices(current_data)
            )
            if not success:
                continue
            self._add_to_data(data, current_data)
        return pd.DataFrame(data)

    def _get_sequence(
        self,
        sequences: Iterable[SpacedSubsequence],
        current_data: _CurrentData,
        fail_main_text: str,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> SpacedSubsequence | None:
        spaced_sequences = list(sequences)
        if len(spaced_sequences) != 1:
            llm = current_data.llm
            logprobs = current_data.logprobs
            logger_name = f"accord.postprocess.{llm}.{self.variant_id}"
            log_file = f"{self.variant_id}-{llm}-{datetime.now().isoformat()}.log"
            log_path = os.path.join(self.logging_dir, log_file)
            self.logger = self.logger or init_logger(logger_name, log_path)
            no_line_breaks = logprobs.to_text(start_idx, end_idx).replace("\n", "<NL>")
            self.logger.info(
                f"AccordID={current_data.get_accord_id()}: {fail_main_text} has "
                f"{len(spaced_sequences)} occurrences in '{no_line_breaks}'."
            )
            return None
        return spaced_sequences[0]

    def _find_question_end_index(self, current_data: _CurrentData) -> bool:
        question = current_data.get_question()
        sequences = current_data.logprobs.indices_of(question)
        seq = self._get_sequence(sequences, current_data, f"Question '{question}'")
        if seq is not None:
            current_data.question_end_index = max(seq.indices)
        return current_data.question_end_index is not None

    def _find_answer_tag_start_index(self, current_data: _CurrentData) -> bool:
        answer_tag = self.accord_loader.surfacer.suffix_surfacer.prefix.strip()
        sequences = current_data.logprobs.indices_of(answer_tag)
        seq = self._get_sequence(sequences, current_data, f"Answer tag '{answer_tag}'")
        if seq is not None:
            current_data.answer_tag_start_index = min(seq.indices)
        return current_data.answer_tag_start_index is not None

    def _fill_logprob_of_answer_choices(self, current_data: _CurrentData) -> bool:
        # Find the start and end index of each label.
        label_indices = {}
        start_idx = current_data.question_end_index
        label_end_idx = current_data.answer_tag_start_index
        for label in current_data.meta_data.answer_choices.keys():
            seqs = current_data.logprobs.indices_of(label, start_idx, label_end_idx)
            seq = self._get_sequence(
                seqs, current_data, f"Label '{label}'", start_idx, label_end_idx
            )
            if seq is None:
                return False
            label_indices[label] = min(seq.indices), max(seq.indices)

        # Find each term relative to its label and the next (tight bounds).
        results = {}
        for label, term in current_data.get_answer_choices().items():
            start_idx = label_indices[label][1]
            end_idx = label_indices.get(chr(ord(label) + 1), [label_end_idx])[0]
            sequences = current_data.logprobs.indices_of(term, start_idx, end_idx)
            sequence = self._get_sequence(
                sequences, current_data, f"Answer Choice '{term}'", start_idx, end_idx
            )
            if sequence is None:
                return False
            results[label] = sequence.to_chosen_logprobs()

        # If all sequences are found, we are good to go.
        current_data.logprob_of_answer_choices = results
        return True

    @staticmethod
    def _add_to_data(all_data: dict, current_data: _CurrentData) -> None:
        surprisals = {}
        for label, lps in current_data.logprob_of_answer_choices.items():
            # Aggregation is over the list of individual tokens making up a
            # label or a choice (not aggregation multiple labels/choices).
            surprisals[label] = current_data.aggregator.aggregate(lps)

        new_data = {
            # Identifiers.
            DFCols.VARIANT_ID.value: current_data.get_variant_id(),
            DFCols.GROUP_ID.value: current_data.get_group_id(),
            DFCols.ACCORD_ID.value: current_data.get_accord_id(),
            DFCols.CSQA_ID.value: current_data.get_csqa_id(),
            DFCols.LLM.value: current_data.llm,
            # Factors.
            DFCols.IS_FACTUAL.value: current_data.is_factual(),
            DFCols.REASONING_HOPS.value: current_data.get_reasoning_hops(),
            DFCols.DISTRACTORS.value: current_data.get_distractors(),
            DFCols.SUBSET.value: current_data.subset.value,
            DFCols.ACCORD_LABEL.value: current_data.get_accord_label(),
            DFCols.CSQA_LABEL.value: current_data.get_csqa_label(),
            # Label Metrics.
            DFCols.SURPRISAL_A.value: surprisals["A"],
            DFCols.SURPRISAL_B.value: surprisals["B"],
            DFCols.SURPRISAL_C.value: surprisals["C"],
            DFCols.SURPRISAL_D.value: surprisals["D"],
            DFCols.SURPRISAL_E.value: surprisals["E"],
        }
        for key, value in new_data.items():
            all_data.setdefault(key, []).append(value)


@command(name="accord.postprocess")
class AccordPostProcess:
    def __init__(
        self,
        path: PathConfig,
        accord: Config,
        accord_variants: VariantsConfig,
        user_templates: UserTemplatesConfig,
    ):
        self.path, self.cfg, self.loaders = path, accord, {}
        self.out_dir = self.cfg.postprocess_dir(self.path.accord_exp_dir)
        for variant_id, variant in accord_variants.variants.items():
            if self._skip(variant_id):
                continue
            self.loaders[variant_id] = VariantLoader(
                variant_id=variant_id,
                accord_loader=AccordLoader(variant, self.path, user_templates),
                logging_dir=self.out_dir,
                flip_logprobs=self.cfg.flip_logprobs,
                aggregator=self.cfg.aggregator,
            )
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.data_by_llm = {}

    def run(self):
        for nickname in self.cfg.llms:
            nickname = flatten(nickname)
            all_data = []
            self.print(f"Postprocessing model={nickname}...")
            llm_output_dir = self.cfg.llm_output_dir(self.path.accord_exp_dir, nickname)
            for walk in walk_files(llm_output_dir):
                inference_path, variant_id = walk.path, walk.no_ext()
                if self._skip(variant_id):
                    continue
                self.print(f"    Processing data for variant={variant_id}...")
                all_data.append(self.loaders[variant_id].load(inference_path, nickname))
                self.print("    Done.")
            self.print("    Collating all processed data...")
            out_dir = self.cfg.postprocess_dir(self.path.accord_exp_dir)
            out_file = ensure_path(os.path.join(out_dir, nickname + ".csv"))
            pd.concat(all_data, ignore_index=True).to_csv(out_file, index=False)
            self.print("    Done.")

    def _skip(self, test_variant_id: VariantID) -> bool:
        for variant_id in self.cfg.collate_variants:
            if test_variant_id == variant_id:
                return False
        return True
