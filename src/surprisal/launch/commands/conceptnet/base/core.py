from dataclasses import dataclass, replace
from typing import Any, Hashable
from enum import Enum
import os
import re

import spacy
from spacy.tokens import Token
from tqdm import tqdm
import pandas as pd

from .....core import SpacedSubsequence

RelationType = str
ConceptNetTerm = str
QueryResult = set[ConceptNetTerm]


@dataclass(frozen=True)
class Triplet:
    source: ConceptNetTerm
    relation: RelationType
    target: ConceptNetTerm


class QueryMethod(Enum):
    """
    The method of ConceptNet triplet instantiation to use. Each method limits the
    choice of variable instantiation in a different way.

    Notes:
    1. All non-factual target variables are chosen from ConceptNet's vocabulary of
       the same relation type, rather than some alternative vocabulary. This limits
       out-of-distribution effects.
    2. We assume that if an assertion is not in ConceptNet, then it is non-factual.
       This is not technically correct, since ConceptNet is not a complete KB of
       factual commonsense assertions. See the ACCORD paper for rationale.

    FACTUAL:
        For a given relation type, r, and source variable instantiation, s, the
        target variable can take on any target value in ConceptNet, v, that *is* an
        existing assertion in ConceptNet.

    NON_FACTUAL:
        For a given relation type, r, and source variable instantiation, s, the
        target variable can take on any target value in ConceptNet, v, that is not an
        existing assertion in ConceptNet, but *does* share the same relation type.
        That is, any v from any factual assertion (s', r, v) in ConceptNet is allowed
        so long as (s, r, v) is not an existing assertion in ConceptNet.
    """

    FACTUAL = "FACTUAL"
    NON_FACTUAL = "NON_FACTUAL"


@dataclass
class Query:
    """
    Contains all information relevant to query ConceptNet.

    relation_type: The RelationType to inform querying.
    source_term: The instantiated Term of the source variable to inform querying.
    method: The method employed in constructing a QueryResult.
    """

    relation_type: RelationType
    source_term: ConceptNetTerm
    method: QueryMethod


@dataclass
class TermComponents:
    language: str
    main: str
    pos: str | None = None

    def matches(
        self,
        other: "TermComponents",
        language: bool = False,
        main: bool = False,
        pos: bool = False,
    ) -> bool:
        if language and self.language != other.language:
            return False
        if main and self.main != other.main:
            return False
        if pos and self.pos != other.pos:
            return False
        return True


class TermFormatter:
    def __init__(self, language: str, cache: bool = False):
        self.has_main_pattern = re.compile("/c/../([^/]+)")
        self.has_pos_pattern = re.compile("/c/../([^/]+)/([^/]+)")
        self.language = language
        self.text_cache = {} if cache else None
        self.comp_cache = {} if cache else None

    @staticmethod
    def _check_cache(cache: dict | None, key: Hashable) -> Any | None:
        return None if cache is None else cache.get(key, None)

    @staticmethod
    def _fill_cache(cache: dict | None, key: Hashable, value: Any) -> Any:
        if cache is not None:
            cache.setdefault(key, value)
        return value

    def ensure_plain_text(self, text: str | ConceptNetTerm) -> str:
        """Formats text (either ConceptNet Term or plain text) into plain text."""
        result = self._check_cache(self.text_cache, text)
        if result is None:
            main = self.get_main_tag(text)
            result = text if main is None else main.replace("_", " ")
        return self._fill_cache(self.text_cache, text, result)

    def get_main_tag(self, text: str | ConceptNetTerm) -> str | None:
        """
        If the text is a Term of the form:
            '/c/<lang>/<main>'
        or
            '/c/<lang>/<main>/<rest-of-hierarchy>'
        extracts the <main> tag.

        Otherwise, returns None.
        """
        match = self.has_main_pattern.match(text)
        if match is None:
            return None  # Likely already plain text.
        return match.group(1)

    def get_pos_tag(self, text: str | ConceptNetTerm) -> str | None:
        """
        If the text is a Term of the form:
            '/c/<lang>/<main>/<pos>'
        or
            '/c/<lang>/<main>/<pos>/<rest-of-hierarchy>'
        extracts the <pos> tag.

        Otherwise, returns None.
        """
        match = self.has_pos_pattern.match(text)
        if match is None:
            return None  # Can't find POS tag.
        return match.group(2)

    def decompose(self, text: str | ConceptNetTerm) -> TermComponents:
        result = self._check_cache(self.comp_cache, text)
        if result is None:
            result = TermComponents(
                language=self.language,
                main=self.get_main_tag(text),
                pos=self.get_pos_tag(text),
            )
        return self._fill_cache(self.comp_cache, text, result)


class ConceptNet:
    """Interface to an underlying monolingual subset of a ConceptNet database."""

    def __init__(self, input_dir: str, formatter: TermFormatter):
        self.formatter = formatter
        self.relation_to_df_map = {}
        for path, _, files in os.walk(input_dir):
            for file in files:
                df = pd.read_csv(os.path.join(path, file), header=None)
                df.columns = [self.source, self.target]
                relation_type = os.path.splitext(file)[0]
                self.relation_to_df_map[relation_type] = df

    def get_all_triplets(self, relation_type: RelationType) -> list[Triplet]:
        result = []
        for data in self.relation_to_df_map[relation_type].to_dict(orient="records"):
            result.append(Triplet(data[self.source], relation_type, data[self.target]))
        return result

    def query(self, query: Query) -> QueryResult:
        """Returns a set of candidate instantiations for a Query."""
        query = replace(query, source_term=query.source_term)
        if query.method == QueryMethod.FACTUAL:
            return self._factual_query(query)
        return self._non_factual_query(query)

    @staticmethod
    def _find_matches(df: pd.DataFrame, col: str, term: ConceptNetTerm) -> pd.DataFrame:
        return df.loc[df[col] == term]
        # Or this:
        # return df[df[col].apply(lambda x: x == term or x.startswith(term + "/"))]

    def _factual_query(self, query: Query) -> QueryResult:
        df = self.relation_to_df_map[query.relation_type]
        df = self._find_matches(df, self.source, query.source_term)
        return set() if df.empty else set(df[self.target].tolist())

    def _non_factual_query(self, query: Query) -> QueryResult:
        df = self.relation_to_df_map[query.relation_type]
        hits = set(df[self.target].tolist())
        factual_blacklist = self._factual_query(query)
        return hits - factual_blacklist

    @property
    def source(self) -> str:
        """Returns the name of the source column in the underlying dataframe."""
        return "source"

    @property
    def target(self) -> str:
        """Returns the name of the target column in the underlying dataframe."""
        return "target"


class ConceptNetFormatter:
    accord_templates = {
        "AtLocation": "Suppose that [{s}] appears near [{t}]",
        "Causes": "Suppose that [{s}] causes [{t}]",
        "HasPrerequisite": "Suppose that [{s}] has prerequisite [{t}]",
        "IsA": "Suppose that [[{s}] is a type of [{t}]",
        "PartOf": "Suppose that [{s}] is a part of [{t}]",
        "UsedFor": "Suppose that [{s}] is used for [{t}]",
    }

    def __init__(self, template: str, formatter: TermFormatter):
        self.template = template
        self.formatter = formatter

    def __call__(self, triplet: Triplet) -> str:
        s = self.formatter.ensure_plain_text(triplet.source)
        t = self.formatter.ensure_plain_text(triplet.target)
        data = self.accord_templates[triplet.relation].format(s=s, t=t)
        return self.template.format(data=data)

    @staticmethod
    def is_desired_target(spaced_sequence: SpacedSubsequence) -> bool:
        is_source_or_target = spaced_sequence.verify("[", "]")
        is_source = spaced_sequence.verify("that [", "]")
        return is_source_or_target and not is_source


@dataclass
class LinguisticFeatures:
    pos: bool = True
    tag: bool = True
    dep: bool = True
    full_morph: bool = True
    children: bool = False


LinguisticsID = str
_MISSING = object()
_INCONSISTENT = "_INCONSISTENT"


@dataclass
class LinguisticsConfig:
    features: dict[LinguisticsID, LinguisticFeatures]


@dataclass
class LinguisticsAnalysis:
    text: str
    pos: str | None
    tag: str | None
    dep: str | None
    morph: dict[str, str] | None
    children: list["LinguisticsAnalysis"]
    raw_term: ConceptNetTerm | None  # Only for caching. Doesn't affect consistency.
    idx: int  # Only needed for caching and doesn't affect consistency.

    @classmethod
    def from_token(
        cls, token: Token, features: LinguisticFeatures, raw_term: ConceptNetTerm | None
    ) -> "LinguisticsAnalysis":
        return LinguisticsAnalysis(
            text=token.text,
            pos=token.pos_ if features.pos else None,
            tag=token.tag_ if features.tag else None,
            dep=token.dep_ if features.dep else None,
            morph=token.morph.to_dict() if features.full_morph else None,
            children=[cls.from_token(c, features, None) for c in token.children],
            raw_term=raw_term,
            idx=token.i,
        )

    def is_consistent_with(
        self,
        other: "LinguisticsAnalysis",
        features: LinguisticFeatures,
        include_top_level_dep: bool = False,
        exclude_text: bool = False,
    ) -> bool:
        # Compare every field other than children.
        consistent = (
            (True if exclude_text else self.text == other.text)
            and self.pos == other.pos
            and self.tag == other.tag
            and (self.dep == other.dep if include_top_level_dep else True)
            and self.morph == other.morph
        )

        # If consistent at top-level, recursively check matching children, if desired.
        if consistent:
            if features.children:
                return self._process_children(other, features, exclude_text)
            return True
        return False

    def _process_children(
        self,
        other: "LinguisticsAnalysis",
        features: LinguisticFeatures,
        exclude_text: bool,
        also_swap: bool = True,
    ) -> bool:
        for c in self.children:
            has_match = False
            for o in other.children:
                # If the number of grand-children don't match, the children won't.
                if len(c.children) != len(o.children):
                    continue
                # If they do match in quantity, recursively check all properties.
                if c.is_consistent_with(o, features, exclude_text=exclude_text):
                    has_match = True
            if not has_match:
                return False
        if also_swap:
            return other._process_children(
                self, features, exclude_text=exclude_text, also_swap=False
            )
        return True

    def matches_concept_net(self, cnet_pos: str | None) -> bool:
        if cnet_pos is None:
            return self.pos is None
        if cnet_pos == "n":
            return self.pos in ["NOUN", "PROPN"]
        if cnet_pos == "v":
            return self.pos == "VERB"
        if cnet_pos == "a":
            return self.pos == "ADJ"
        if cnet_pos == "r":
            return self.pos == "ADV"
        return False


class LinguisticsAnalyzer:
    def __init__(
        self,
        features: LinguisticFeatures,
        formatter: ConceptNetFormatter,
        verbose: bool,
    ):
        self.features = features
        self.formatter = formatter
        self.verbose = verbose
        self.nlp = spacy.load("en_core_web_lg")
        self.dep_caches: dict[ConceptNetTerm, dict[Triplet, str | None]] = {}
        self.analysis_cache: dict[ConceptNetTerm, LinguisticsAnalysis] = {}
        self.consistency_cache: dict[tuple[ConceptNetTerm, ConceptNetTerm], bool] = {}

    def validate_targets(self, factual_triplets: list[Triplet]) -> None:
        """
        A target is only valid if it's contextual DEP tag is consistent across all
        triplets in which it occurs (as well as being self-consistent across the same).
        """
        for triplet in tqdm(factual_triplets) if self.verbose else factual_triplets:
            self._cache_target(triplet)
        for target, raw_analysis in self.analysis_cache.items():
            deps = set(self.dep_caches[target].values())
            if len(deps) == 1:
                raw_analysis.dep = deps.pop()
            else:
                raw_analysis.dep = _INCONSISTENT

    def _cache_target(self, triplet: Triplet) -> None:
        # Get the raw analysis, either from cache or by computing.
        raw_analysis = self.analysis_cache.get(triplet.target, _MISSING)
        if raw_analysis is _MISSING:
            raw_analysis = self._analyze_raw(triplet.target)
            self.analysis_cache[triplet.target] = raw_analysis

        # Ensure the dependency sub-cache is lacking this particular triplet.
        dep_cache = self.dep_caches.setdefault(triplet.target, {})
        if dep_cache.get(triplet, _MISSING) is not _MISSING:
            raise ValueError(f"Triplet is not unique in ConceptNet: {triplet}")

        # Record the dependency tag if this triplet is self-consistent. Else None.
        contextual_analysis = self._analyze_contextual(raw_analysis.idx, triplet)
        consistent = raw_analysis.is_consistent_with(contextual_analysis, self.features)
        dep_cache[triplet] = contextual_analysis.dep if consistent else _INCONSISTENT

    def is_valid_factual(self, target: ConceptNetTerm) -> bool:
        """Raises KeyError if target has not been preprocessed."""
        return self.analysis_cache[target].dep != _INCONSISTENT

    def is_valid_non_factual(
        self,
        target: ConceptNetTerm,
        factual_target: ConceptNetTerm,
        concept_net_pos: str,
    ) -> bool:
        """
        Returns True only if this non-factual triplet's analysis is both consistent
        with the analysis of the factual target and the ConceptNet components' POS
        (if LinguisticFeatures permits). Non-factual contextual consistency is not
        checked in order to leverage caching; however, factual contextual consistency
        has been previously validated.

        Raises KeyError if either target has not been (factually) validated.
        """
        # Retrieve the consistency result, if any.
        value = self.consistency_cache.get((target, factual_target), None)
        if value is not None:
            return value

        analysis = self.analysis_cache[target]
        factual_analysis = self.analysis_cache[factual_target]

        if _INCONSISTENT in [analysis.dep, factual_analysis.dep]:
            # If either analysis is not (factually) self-consistent, then it's no good.
            value = False
        elif self.features.pos and not analysis.matches_concept_net(concept_net_pos):
            # If the target analysis is not consistent with ConceptNet, then no good.
            value = False
        else:
            # Determine consistency between target and factual.
            value = analysis.is_consistent_with(
                factual_analysis,
                features=self.features,
                include_top_level_dep=True,
                exclude_text=True,
            )

        # Cache and return.
        self.consistency_cache[(target, factual_target)] = value
        return value

    def _analyze_raw(self, target: ConceptNetTerm) -> LinguisticsAnalysis:
        for tok in self.nlp(self.formatter.formatter.ensure_plain_text(target)):
            if tok.dep_ == "ROOT":  # "ROOT" is baked into spacy.
                return LinguisticsAnalysis.from_token(tok, self.features, target)
        raise ValueError(f"Spacy cannot determine a ROOT for: {target}")

    def _analyze_contextual(self, target_idx: int, t: Triplet) -> LinguisticsAnalysis:
        context = self.formatter(t)
        doc = self.nlp(context)
        last_bracket_index = None
        for token in doc:
            if token.text.strip() == "[":
                last_bracket_index = token.i
        if last_bracket_index is None:
            raise ValueError(f"Couldn't find '[' token in: {context}")
        for token in doc:
            if token.i - last_bracket_index - 1 == target_idx:
                return LinguisticsAnalysis.from_token(token, self.features, t.target)
