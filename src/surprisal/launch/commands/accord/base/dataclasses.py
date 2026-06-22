from dataclasses import dataclass

Label = str
Term = str
Relation = str

AccordID = str
CaseID = int
StatementID = str
CsqaID = str

StatementKey = tuple[Term, Term, Relation]


@dataclass
class Pairing:
    """
    Dataclass to store pairing data.

    Fields:
        id: Identifier of the pairing statement (which has the same ID in each tree)
        flip_negative: Whether the positive/negative variation logic needs to be flipped
        relation_type: The skill type of the relation of the pairing statement.
    """

    id: StatementID
    flip_negative: bool
    relation_type: Relation


@dataclass
class Statement:
    """
    Dataclass to store statement data.

    Fields:
        surface_form: Format string for the surface (i.e., text) form of the statement
        source_term: String value of the source term
        target_term: String value of the target term
        relation_type: The skill type of the relation of the statement
    """

    surface_form: str
    source_term: Term
    target_term: Term
    relation_type: Relation

    def to_key(self) -> StatementKey:
        return self.source_term, self.target_term, self.relation_type


@dataclass
class MetaData:
    """
    Dataclass to store all instance meta-data.

    Fields:
        id: Identifier for this instance within the data subset
        qa_id: Identifier of the CSQA instance from the base dataset
        question: Question text of the CSQA instance
        answer_choices: Answer choices of the CSQA instance; form: {label: term}
        label: Chosen answer label (which can be different from the CSQA instance label)
        pairing: Pairing data
        reduction_cases: Indicates which statements reduce to which cases wrt the
            pairing statement; form {statement_id: case_id}
        statements: All Statement data for each statement in each tree; form:
            {answer_label/tree_label: {statement_id: Statement}}
        statements_order: Ordered list of all Statements from all trees; form:
            [(answer_label/tree_label, statement_id)]
    """

    id: AccordID
    qa_id: CsqaID
    question: str
    answer_choices: dict[Label, Term]
    label: Label
    pairing: Pairing | None
    reduction_cases: dict[StatementID, CaseID] | None
    statements: dict[Label, dict[StatementID, Statement]]
    statements_ordering: list[list]

    def is_factual(self, csqa_label: Label) -> bool:
        return csqa_label == self.label

    def is_baseline(self) -> bool:
        return self.reduction_cases is None

    def get_statement(self, ordering: tuple[Label, StatementID]) -> Statement:
        """Returns a specific statement based on an ordering tuple entry."""
        return self.statements[ordering[0]][ordering[1]]


@dataclass
class Instance:
    """
    Dataclass to store all ACCORD instance data.

    Fields:
        text: Fully surfaced text string of the instance data for LLM consumption
        csqa_label: Ground truth label of the base CSQA instance from which this ACCORD
            derives (which can be different from the ACCORD's chosen answer label)
        meta_data: Metadata for this ACCORD instance
    """

    text: str
    csqa_label: Label
    meta_data: MetaData

    def is_factual(self) -> bool:
        return self.meta_data.is_factual(self.csqa_label)

    def get_factuality_independent_group_id(self) -> str:
        return self.meta_data.id[:-2]


@dataclass
class CaseLink:
    """
    A permutation case linking two statements based on their relations' skill type.
    Typically, one statement will dominate over the other.

    The five permutation cases are:
        Case 0: A relation1 B; C relation2 D (no linking relation)
        Case 1: A relation1 B; B relation2 C (relation1.target == relation2.source)
        Case 2: A relation1 B; C relation2 B (relation1.target == relation2.target)
        Case 3: B relation1 A; B relation2 C (relation1.source == relation2.source)
        Case 4: B relation1 A; C relation2 B (relation1.source == relation2.target)

    NOTE: The cases are equivalent with respect to permutation of the order of the
    statements unless the relation types are the same (where the surface form of the
    subsumed statement can sometimes be different). In general, however:
        Case 1 permutes to case 4
        Case 4 permutes to case 1
        All other cases permute to themselves

    Fields:
        r1_type: Relation skill type of the dominant statement
        r2_type: Relation skill type of the subsumed statement
        case: Permutation case linking the two statements
    """

    r1_type: Relation
    r2_type: Relation
    case: CaseID

    def equivalent(self) -> "CaseLink":
        if self.case == 1:
            return CaseLink(self.r2_type, self.r1_type, 4)
        elif self.case == 0 or self.case == 2 or self.case == 3:
            return CaseLink(self.r2_type, self.r1_type, self.case)
        elif self.case == 4:
            return CaseLink(self.r2_type, self.r1_type, 1)
        else:
            raise TypeError(f"Unsupported permutation case: {self.case}")

    def as_tuple(self) -> tuple[Relation, Relation, CaseID]:
        return self.r1_type, self.r2_type, self.case


class SurfaceForms:
    """
    Keeps track of surface form variations of subsumed relations in a CaseLink.
    """

    def __init__(self):
        self.permutations = {}

    def register(self, case_link: CaseLink, subsumed_surface_form: str):
        """Registers a CaseLink and its associated subsumed surface form."""
        self.permutations[case_link.as_tuple()] = subsumed_surface_form

    def get(self, case_link: CaseLink) -> str | None:
        """
        Returns the surface form for a CaseLink. Returns the surface form of the
        equivalent CaseLink under case permutation if no surface form is registered
        for the given CaseLink. Returns None if neither the given nor the equivalent
        CaseLink have registered surface forms.
        """
        equiv = case_link.equivalent().as_tuple()
        if case_link.as_tuple() in self.permutations:
            return self.permutations[case_link.as_tuple()]
        elif equiv in self.permutations:
            return self.permutations[equiv]
        else:
            return None


@dataclass
class CsqaBase:
    """
    Dataclass to store all data for an instance of base CSQA.

    Fields:
        identifier: Identifier of the CSQA instance from the base dataset
        question: Question text of the CSQA instance
        correct_answer_label: Ground truth answer label if the CSQA instance
        answer_choices: Answer choices of the CSQA instance; form: {label: term}
        pairing_templates: ignored
    """

    identifier: CsqaID
    question: str
    correct_answer_label: Label
    answer_choices: dict[Label, Term]
    pairing_templates: list[dict]
