from dataclasses import dataclass, field
from random import Random
from typing import Any
from enum import Enum

from .core import ConceptNetTerm, QueryResult, TermComponents, TermFormatter

RankerID = str
RankMetric = Any
RankItem = tuple[ConceptNetTerm, RankMetric | None]
Ranking = list[RankItem]
Seed = None | int | float | str | bytes | bytearray


class RankingMethod(Enum):
    NOOP = "NOOP"
    RANDOM = "RANDOM"
    CNET_HIERARCHY = "CNET_HIERARCHY"


class Ranker:
    def __call__(self, result: QueryResult, *args, **kwargs) -> Ranking:
        raise NotImplementedError

    def is_worst_outcome(self, metric: RankMetric | None) -> bool:
        """
        Returns whether the given metric is the worst or nearly the worst possible
        outcome. If the top-ranked item has this metric value, it suggests that no
        candidate item was a good fit according to this ranker.
        """
        raise NotImplementedError

    def is_likely_low_ranked(self, *args, **kwargs) -> bool:
        """Returns whether a term would likely receive a low rank."""
        raise NotImplementedError


class NoopRanker(Ranker):
    def __call__(self, result: QueryResult, *args, **kwargs) -> Ranking:
        return [(term, None) for term in result]

    def is_worst_outcome(self, _: RankMetric | None) -> bool:
        return False  # No unranked outcome is worse than any other.

    def is_likely_low_ranked(self, *args, **kwargs) -> bool:
        return False  # No unranked "ranking" is worse than any other.


class RandomRanker(Ranker):
    def __init__(self, seed: Seed = None):
        self.random = Random(seed)

    def __call__(self, result: QueryResult, *args, **kwargs) -> Ranking:
        result = [(term, None) for term in result]
        self.random.shuffle(result)  # Shuffles in place and returns None.
        return result

    def is_worst_outcome(self, _: RankMetric | None) -> bool:
        return False  # No random outcome is worse than any other, on average.

    def is_likely_low_ranked(self, *args, **kwargs) -> bool:
        return False  # No random ranking is worse than any other, on average.


class ConceptNetHierarchyRanker(Ranker):
    def __init__(
        self,
        language: bool = False,
        main: bool = False,
        pos: bool = False,
        shuffle: bool = True,
        seed: Seed = None,
    ):
        self.kwargs = dict(language=language, main=main, pos=pos)
        self.ranker = RandomRanker(seed) if shuffle else None

    def __call__(
        self,
        result: QueryResult,
        *args,
        formatter: TermFormatter = None,
        factual_target: ConceptNetTerm = None,
        **kwargs,
    ) -> Ranking:
        """
        Checks whether the query result term has the same selected hierarchy tags as
        the factual term (based on the tags selected in __init__). If so, these are
        ranked first (optionally shuffled), before any terms where the hierarchy is
        mismatched (also optionally shuffled).

        The hierarchy check is strict: both terms must have the same selected tags to
        match. If some or all tags are missing from one term or the other (because the
        two terms are at different levels of the ConceptNet hierarchy), they are
        considered non-matching.

        Necessary kwargs:
        - a TermFormatter able to extract the tags from terms
        - the factual target term corresponding to the given QueryResult
        """
        match, no_match = set(), set()
        for term in result:
            factual_comps = formatter.decompose(factual_target)
            term_comps = formatter.decompose(term)
            if factual_comps.matches(term_comps, **self.kwargs):
                match.add(term)
            else:
                no_match.add(term)
        if self.ranker:
            match = [(t, 1) for t, _ in self.ranker(match, *args, **kwargs)]
            no_match = [(t, 0) for t, _ in self.ranker(no_match, *args, **kwargs)]
        else:
            match, no_match = [(t, 1) for t in match], [(t, 0) for t in no_match]
        return match + no_match

    def is_worst_outcome(self, metric: RankMetric | None) -> bool:
        return metric == 0

    def is_likely_low_ranked(
        self, *args, main: TermComponents, comparison: TermComponents = None, **kwargs
    ) -> bool:
        """
        Formatter is mandatory. Comparison is optional. If non-None, returns whether
        the term and the comparison match hierarchically. If None, returns True if
        the term lacks a POS tag and pos=True in __init__. Returns False otherwise.
        """
        if comparison is not None:
            return not comparison.matches(main, **self.kwargs)
        return self.kwargs["pos"] and main.pos is None


@dataclass
class RankerInfo:
    method: RankingMethod
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankersConfig:
    rankers: dict[RankerID, RankerInfo]


def ranker_factory(method: RankingMethod, **data) -> Ranker:
    if method == RankingMethod.NOOP:
        return NoopRanker()
    elif method == RankingMethod.RANDOM:
        return RandomRanker(**data)
    elif method == RankingMethod.CNET_HIERARCHY:
        return ConceptNetHierarchyRanker(**data)
    else:
        raise ValueError(f"Unsupported ranking method: {method}")


class RankerManager:
    def __init__(self, cfg: RankersConfig):
        self.rankers = {}
        for ranker_id, info in cfg.rankers.items():
            self.rankers[ranker_id] = ranker_factory(info.method, **info.data)

    def get(self, ranker_id: RankerID) -> Ranker:
        try:
            return self.rankers[ranker_id]
        except KeyError:
            raise ValueError(f"Missing ranker data for: '{ranker_id}'.")
