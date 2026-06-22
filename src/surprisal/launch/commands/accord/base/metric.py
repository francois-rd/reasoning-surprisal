from typing import Iterable, Hashable
from dataclasses import dataclass
from enum import Enum


from .....core import AggregatorOption, AggregatorStr

from .dataclasses import Label


class AbsMetricType(Enum):
    LABEL = "LABEL"
    CHOICE = "CHOICE"


class AbsMetricSubType(Enum):
    MATCHING_ACCORD = "MATCHING_ACCORD"
    MATCHING_CSQA = "MATCHING_CSQA"
    ALL = "ALL"


@dataclass
class AbsMetricID:
    metric: AbsMetricType
    sub_metric: AbsMetricSubType
    agg: AggregatorOption

    @staticmethod
    def yield_all(aggregators: list[AggregatorOption]) -> Iterable["AbsMetricID"]:
        for metric in AbsMetricType:
            for sub_metric in AbsMetricSubType:
                for agg in aggregators:
                    yield AbsMetricID(metric, sub_metric, agg)


@dataclass
class PredictionLogprobs:
    label_lps: dict[Label, dict[AggregatorStr, float]]
    choice_lps: dict[Label, dict[AggregatorStr, float]]


@dataclass
class AbsoluteMetrics:
    label_matching_accord: dict[AggregatorStr, float]
    label_matching_csqa: dict[AggregatorStr, float]
    label_all: dict[AggregatorStr, float]

    choice_matching_accord: dict[AggregatorStr, float]
    choice_matching_csqa: dict[AggregatorStr, float]
    choice_all: dict[AggregatorStr, float]

    predict_lps: PredictionLogprobs

    @staticmethod
    def as_attribute_name(metric_id: AbsMetricID) -> str:
        return "_".join([metric_id.metric.value, metric_id.sub_metric.value]).lower()

    def get(self, metric_id: AbsMetricID) -> float | int:
        return getattr(self, self.as_attribute_name(metric_id))[metric_id.agg.value]

    @classmethod
    def from_data(
        cls,
        label_lps: dict[Label, dict[AggregatorOption, float]],
        choice_lps: dict[Label, dict[AggregatorOption, float]],
        accord_label: Label,
        csqa_label: Label,
    ) -> "AbsoluteMetrics":
        return AbsoluteMetrics(
            # Label x accord/csqa/all surprisal.
            label_matching_accord={
                agg.value: lp for agg, lp in label_lps[accord_label].items()
            },
            label_matching_csqa={
                agg.value: lp for agg, lp in label_lps[csqa_label].items()
            },
            label_all=cls._surprisal_all(label_lps),
            # Label x accord/csqa/all surprisal.
            choice_matching_accord={
                agg.value: lp for agg, lp in choice_lps[accord_label].items()
            },
            choice_matching_csqa={
                agg.value: lp for agg, lp in choice_lps[csqa_label].items()
            },
            choice_all=cls._surprisal_all(choice_lps),
            # Predict logprobs.
            predict_lps=PredictionLogprobs(
                label_lps=cls._predict_lps(label_lps),
                choice_lps=cls._predict_lps(choice_lps),
            ),
        )

    @classmethod
    def _surprisal_all(
        cls, data: dict[Label, dict[AggregatorOption, float]]
    ) -> dict[AggregatorStr, float]:
        data_by_agg = {}
        for label, lp_by_agg in data.items():
            for agg, lp in lp_by_agg.items():
                data_by_agg.setdefault(agg, []).append(lp)
        return {agg.value: agg.aggregate(lp) for agg, lp in data_by_agg.items()}

    @staticmethod
    def _predict_lps(
        data: dict[Label, dict[AggregatorOption, float]],
    ) -> dict[Label, dict[AggregatorStr, float]]:
        return {
            label: {agg.value: lp for agg, lp in label_data.items()}
            for label, label_data in data.items()
        }


class RelativeMetricType(Enum):
    LABEL = "LABEL"
    CHOICE = "CHOICE"


@dataclass
class RelativeMetricID:
    metric: RelativeMetricType
    agg: AggregatorOption

    @staticmethod
    def yield_all(aggregators: list[AggregatorOption]) -> Iterable["RelativeMetricID"]:
        for metric in RelativeMetricType:
            for agg in aggregators:
                yield RelativeMetricID(metric, agg)


@dataclass
class RelativeMetrics:
    label: dict[AggregatorStr, float]
    choice: dict[AggregatorStr, float]
    predict_lps: PredictionLogprobs

    @staticmethod
    def as_attribute_name(metric_id: RelativeMetricID) -> str:
        return f"relative_{metric_id.metric.value.lower()}"

    def get(self, metric_id: RelativeMetricID) -> float:
        return getattr(self, self.as_attribute_name(metric_id))[metric_id.agg.value]

    @classmethod
    def from_data(
        cls,
        factual_label_lps: dict[Label, dict[AggregatorOption, float]],
        factual_choice_lps: dict[Label, dict[AggregatorOption, float]],
        non_factual_label_lps: dict[Label, dict[AggregatorOption, float]],
        non_factual_choice_lps: dict[Label, dict[AggregatorOption, float]],
    ) -> "RelativeMetrics":
        return RelativeMetrics(
            label=cls._agg(cls._pair_up(factual_label_lps, non_factual_label_lps)),
            choice=cls._agg(cls._pair_up(factual_choice_lps, non_factual_choice_lps)),
            predict_lps=PredictionLogprobs(
                label_lps=cls._lps(factual_label_lps, non_factual_label_lps),
                choice_lps=cls._lps(factual_choice_lps, non_factual_choice_lps),
            ),
        )

    @staticmethod
    def _pair_up(
        factual_or_correct_lps: dict[Hashable, dict[AggregatorOption, float]],
        af_or_incorrect_lps: dict[Hashable, dict[AggregatorOption, float]],
    ) -> dict[AggregatorOption, list[float]]:
        result = {}
        for key, f_or_c_data in factual_or_correct_lps.items():
            for agg, af_or_i_lp in af_or_incorrect_lps[key].items():
                result.setdefault(agg, []).append(f_or_c_data[agg] - af_or_i_lp)
        return result

    @staticmethod
    def _agg(data: dict[AggregatorOption, list[float]]) -> dict[AggregatorStr, float]:
        return {agg.value: agg.aggregate(lp) for agg, lp in data.items()}

    @staticmethod
    def _lps(
        factual_or_correct_lps: dict[Label, dict[AggregatorOption, float]],
        af_or_incorrect_lps: dict[Label, dict[AggregatorOption, float]],
    ) -> dict[Label, dict[AggregatorStr, float]]:
        result = {}
        for key, f_or_c_data in factual_or_correct_lps.items():
            for agg, af_or_i_lp in af_or_incorrect_lps[key].items():
                result.setdefault(key, {})[agg.value] = f_or_c_data[agg] - af_or_i_lp
        return result
