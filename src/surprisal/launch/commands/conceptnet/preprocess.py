from random import Random
import os

from coma import command
from tqdm import tqdm
import spacy

from ....core import UserTemplatesConfig
from ....io import ConditionalPrinter, PathConfig, save_dataclass_jsonl

from .base import (
    ConceptNet,
    ConceptNetFormatter,
    ConceptNetTerm,
    LinguisticsAnalyzer,
    LinguisticsConfig,
    Query,
    QueryMethod,
    TermFormatter,
    Triplet,
    Ranker,
    RankerManager,
    RankersConfig,
)

from .base import Config, TripletVariantCluster, VariantID, VariantsConfig


@command(name="cnet.preprocess")
class ConceptNetPreprocess:
    def __init__(
        self,
        path: PathConfig,
        conceptnet: Config,
        cnet_variants: VariantsConfig,
        rankers: RankersConfig,
        linguistics: LinguisticsConfig,
        user_templates: UserTemplatesConfig,
    ):
        self.path = path
        self.cfg = conceptnet
        self.variants = cnet_variants.variants
        self.linguistics = linguistics
        self.user_templates = user_templates
        self.formatter = TermFormatter(language="en")
        self.concept_net = ConceptNet(path.concept_net_dir, self.formatter)
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.stopwords = spacy.load("en_core_web_lg").Defaults.stop_words
        self.analyzers = self._create_analyzers()
        self.rankers = self._create_rankers(RankerManager(rankers))

    def _create_analyzers(self) -> dict[VariantID, LinguisticsAnalyzer]:
        analyzers = {}
        for variant_id, variant in self.variants.items():
            t = self.user_templates.templates[variant.user_template_id].template
            analyzer = LinguisticsAnalyzer(
                features=self.linguistics.features[variant.linguistics_id],
                formatter=ConceptNetFormatter(template=t, formatter=self.formatter),
                verbose=self.cfg.verbose,
            )
            analyzers[variant_id] = analyzer
        return analyzers

    def _create_rankers(self, manager: RankerManager) -> dict[VariantID, Ranker]:
        return {v_id: manager.get(v.ranker_id) for v_id, v in self.variants.items()}

    def run(self):
        self.print(f"Processing relation_type={self.cfg.relation_type}...")
        all_triplets = self.concept_net.get_all_triplets(self.cfg.relation_type)
        for variant_id, variant in self.variants.items():
            self.print(f"    Pre-screening all triplets for {variant_id=}...")
            self.analyzers[variant_id].validate_targets(all_triplets)
            self.print("    Done.")
        self.print("    Creating clusters...")
        triplet_clusters = self._create_clusters(all_triplets)
        self.print("    Done.")
        self.print("    Main processing...")
        self._do_run(triplet_clusters, all_triplets)
        self.print("    Done.")
        self.print("    Saving...")
        self._validate_and_save(triplet_clusters)
        self.print("    Done.")
        self.print("Done.")

    def _create_clusters(
        self, all_triplets: list[Triplet]
    ) -> dict[Triplet, TripletVariantCluster]:
        triplet_clusters = {}
        for triplet in all_triplets:
            if self._fast_skip(triplet, self.cfg.factual_variant_id):
                continue
            cluster = TripletVariantCluster(
                factual_query=Query(
                    relation_type=triplet.relation,
                    source_term=triplet.source,
                    method=QueryMethod.FACTUAL,
                ),
                factual_target=triplet.target,
            )
            triplet_clusters[triplet] = cluster
        return triplet_clusters

    def _do_run(
        self,
        triplet_clusters: dict[Triplet, TripletVariantCluster],
        triplets: list[Triplet],
    ) -> None:
        Random(self.cfg.preprocess_seed).shuffle(triplets)  # Shuffles in place.
        sample_count = 0
        for triplet in tqdm(triplets) if self.cfg.verbose else triplets:
            if sample_count >= self.cfg.subsampling_per_relation:
                break  # Got enough data for this relation type.
            success = True
            for variant_id in self.variants.keys():
                if variant_id == self.cfg.factual_variant_id:
                    continue
                success = success and self._run_for_variant(
                    triplet_clusters, triplet, variant_id
                )
            if success:  # If all variants are successful, increment sample count.
                sample_count += 1

    def _run_for_variant(
        self,
        triplet_clusters: dict[Triplet, TripletVariantCluster],
        triplet: Triplet,
        v_id: VariantID,
    ) -> bool:
        if self._fast_skip(triplet, v_id):
            return False
        query = Query(
            relation_type=triplet.relation,
            source_term=triplet.source,
            method=QueryMethod.NON_FACTUAL,
        )
        candidates = self._get_candidates(query, triplet, v_id)
        if len(candidates) < self.cfg.preprocess_nf_threshold:
            return False
        all_candidates = triplet_clusters[triplet].non_factual_candidates
        if v_id in all_candidates:
            raise ValueError(f"Triplet {triplet} is non-unique in ConceptNet")
        all_candidates[v_id] = candidates
        return True

    def _fast_skip(self, triplet: Triplet, variant_id: VariantID):
        target = self.formatter.ensure_plain_text(triplet.target)
        if self._textual_skip(triplet.source, target):
            return True
        target_comp = self.formatter.decompose(triplet.target)
        if self.rankers[variant_id].is_likely_low_ranked(main=target_comp):
            return True
        return False

    def _textual_skip(self, main_text: str, check_match_text: str) -> bool:
        if any(char.isdigit() for char in main_text):
            # Skip the chemical compound names, etc., that contain digits.
            return True
        if not main_text.isascii():
            # Skip any terms with at least 1 non-ASCII character, since these don't
            # get transmitted well to the vLLM server and back.
            return True
        main_text = self.formatter.ensure_plain_text(main_text)
        if main_text in self.stopwords:
            # Skipping stopwords.
            return True
        if check_match_text is not None:
            # If the two texts match, skip.
            if main_text == check_match_text:
                return True
        return False

    def _get_candidates(
        self, query: Query, factual_triplet: Triplet, variant_id: VariantID
    ) -> list[ConceptNetTerm]:
        if not self.analyzers[variant_id].is_valid_factual(factual_triplet.target):
            # This means the factual target is not linguistically self-consistent
            # under the linguistic features of this variant.
            return []
        candidates = []
        factual_source_text = self.formatter.ensure_plain_text(factual_triplet.source)
        for term in self.concept_net.query(query):
            term_comps = self.formatter.decompose(term)
            if self._textual_skip(term, factual_source_text):
                continue
            ranker_skip = self.rankers[variant_id].is_likely_low_ranked(
                main=term_comps,
                comparison=self.formatter.decompose(factual_triplet.target),
            )
            if ranker_skip:
                continue
            valid = self.analyzers[variant_id].is_valid_non_factual(
                target=term,
                factual_target=factual_triplet.target,
                concept_net_pos=term_comps.pos,
            )
            if valid:
                candidates.append(term)
        return candidates

    def _validate_and_save(
        self, triplet_clusters: dict[Triplet, TripletVariantCluster]
    ) -> None:
        # Split clusters by relation while checking validity.
        valid_clusters = []
        for c in triplet_clusters.values():
            # Ensure all non-factual variants are included in candidates.
            missing_variant = False
            for v_id in self.variants.keys():
                if v_id == self.cfg.factual_variant_id:
                    continue
                if v_id not in c.non_factual_candidates:
                    missing_variant = True
            if missing_variant:
                continue
            valid_clusters.append(c)

        out_dir = self.cfg.preprocess_dir(self.path.cnet_exp_dir)
        out_file = os.path.join(out_dir, f"{self.cfg.relation_type}.jsonl")
        save_dataclass_jsonl(out_file, *valid_clusters, ensure_ascii=False)
