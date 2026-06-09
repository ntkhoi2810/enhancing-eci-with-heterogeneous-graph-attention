"""
Data loading, preprocessing, and collation for Event Causal Identification.

This module handles:
- Loading raw pickle datasets (ESC, CTB, etc.)
- Preprocessing: event tagging, masking, and syntax graph extraction
- Custom collation for batching graph + text data together
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import spacy
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


# ---------------------------------------------------------------------------
# spaCy model (lazy-loaded for syntax parsing)
# ---------------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Please install it:\n"
        "  python -m spacy download en_core_web_sm"
    )

# ---------------------------------------------------------------------------
# Relation types used by the Heterogeneous Graph
# ---------------------------------------------------------------------------
KNOWN_RELATIONS = frozenset({"nsubj", "prep", "pobj", "dobj", "amod", "ROOT"})


# ---------------------------------------------------------------------------
# Syntax graph extraction
# ---------------------------------------------------------------------------
def extract_syntax_graph(
    sentence: str, e1: str, e2: str
) -> Tuple[Dict[str, List[List[int]]], int, int]:
    """Parse a sentence with spaCy and build a heterogeneous dependency graph.

    Each dependency relation becomes an edge type.  Relations not in
    ``KNOWN_RELATIONS`` are collapsed into ``'other'``.

    Args:
        sentence: The raw sentence text.
        e1: Surface form of the first event mention.
        e2: Surface form of the second event mention.

    Returns:
        A tuple of ``(edges, e1_idx, e2_idx)`` where *edges* maps relation
        names to ``[[src_indices], [tgt_indices]]`` and the indices point to
        the token positions of the two events.
    """
    doc = nlp(sentence)
    edges: Dict[str, List[List[int]]] = {}
    e1_idx, e2_idx = 0, 0

    for token in doc:
        if e1.lower() in token.text.lower():
            e1_idx = token.i
        if e2.lower() in token.text.lower():
            e2_idx = token.i

        rel = token.dep_ if token.dep_ in KNOWN_RELATIONS else "other"

        if rel not in edges:
            edges[rel] = [[], []]

        edges[rel][0].append(token.head.i)  # source node
        edges[rel][1].append(token.i)       # target node

    return edges, e1_idx, e2_idx


# ---------------------------------------------------------------------------
# Row-level preprocessing
# ---------------------------------------------------------------------------
def _ireplace(text: str, old: str, new: str) -> str:
    """Case-insensitive single replacement."""
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    return pattern.sub(new, text, count=1)


def preprocess_row(row: pd.Series) -> pd.Series:
    """Transform a raw data row into tagged/masked sentences + syntax graph.

    Produces six columns:
    ``event_tagged_sentence``, ``event_masked_sentence``, ``labels``,
    ``syntax_edges``, ``e1_idx``, ``e2_idx``.
    """
    sent = str(row["sentence"])
    e1 = str(row["e1"])
    e2 = str(row["e2"])

    # Event-tagged sentence
    tagged = _ireplace(sent, e1, f"<e1>{e1}</e1>")
    tagged = _ireplace(tagged, e2, f"<e2>{e2}</e2>")

    # Masked sentence (randomly mask one event)
    to_mask = e1 if random.choice([True, False]) else e2
    masked = _ireplace(sent, to_mask, "<mask>")

    if "<mask>" not in masked:
        other_event = e2 if to_mask == e1 else e1
        masked = _ireplace(sent, other_event, "<mask>")

    if "<mask>" not in masked:
        masked = sent + " <mask>"

    label_id = 1 if row["label_str"] == "causal" else 0

    # Syntax graph
    syntax_edges, e1_idx, e2_idx = extract_syntax_graph(sent, e1, e2)

    return pd.Series([tagged, masked, label_id, syntax_edges, e1_idx, e2_idx])


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_and_preprocess_data(dataset_path: str) -> Dataset:
    """Load a pickle dataset and apply preprocessing.

    Args:
        dataset_path: Path to a ``.pkl`` file with columns
            ``[id, sentence, e1, e2, label_str]``.

    Returns:
        A HuggingFace ``Dataset`` ready for tokenization.
    """
    data = pd.read_pickle(dataset_path)
    df = pd.DataFrame(data)
    df.columns = ["id", "sentence", "e1", "e2", "label_str"]

    df[
        [
            "event_tagged_sentence",
            "event_masked_sentence",
            "labels",
            "syntax_edges",
            "e1_idx",
            "e2_idx",
        ]
    ] = df.apply(preprocess_row, axis=1)

    df = df[
        [
            "sentence",
            "event_tagged_sentence",
            "event_masked_sentence",
            "e1",
            "e2",
            "labels",
            "syntax_edges",
            "e1_idx",
            "e2_idx",
        ]
    ]

    return Dataset.from_pandas(df)


# ---------------------------------------------------------------------------
# Custom collate function (graph + text)
# ---------------------------------------------------------------------------
class GraphTextCollate:
    """Collate function that handles both text tokens and graph structures.

    Standard ``DataCollatorWithPadding`` cannot process the graph fields
    (``syntax_edges``, ``e1_idx``, ``e2_idx``).  This collator separates
    them, pads text normally, and re-attaches the graph data to the batch.
    """

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.text_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        graph_features = {
            "edges": [f["syntax_edges"] for f in features],
            "e1_idx": [f["e1_idx"] for f in features],
            "e2_idx": [f["e2_idx"] for f in features],
        }

        text_features = [
            {k: v for k, v in f.items() if k not in ("syntax_edges", "e1_idx", "e2_idx")}
            for f in features
        ]

        batch = self.text_collator(text_features)
        batch["graph_data"] = graph_features
        return batch