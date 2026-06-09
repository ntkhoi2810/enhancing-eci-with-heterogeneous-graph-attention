"""
Neural network models for Event Causal Identification.

Architecture overview:
    ClozeAnalyzer  — Fills masked events via soft cloze-test on RoBERTa
    Discriminator  — Cross-attention classifier over event representations
    CausalModel    — Full pipeline: Cloze + HAN graph fusion + Discriminator
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import HANConv
from transformers import AutoModelForMaskedLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Relation metadata shared between model and data pipeline
# ---------------------------------------------------------------------------
HAN_METADATA = (
    ["word"],
    [
        ("word", "nsubj", "word"),
        ("word", "prep", "word"),
        ("word", "pobj", "word"),
        ("word", "dobj", "word"),
        ("word", "amod", "word"),
        ("word", "ROOT", "word"),
        ("word", "other", "word"),
    ],
)


class ClozeAnalyzer(nn.Module):
    """Soft cloze-test module that predicts a masked event token.

    Instead of hard-replacing the mask with the argmax prediction, this
    module uses a *soft* embedding (weighted sum over the vocabulary) so
    that gradients can flow back through the prediction step.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        bert: AutoModelForMaskedLM,
        device: torch.device,
        visualize: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = bert
        self.visualize = visualize
        self.device = device

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        groundtruth: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run soft cloze prediction and return the contextual embedding at
        the masked position.

        Args:
            x: Tokenised masked input (``input_ids``, ``attention_mask``).
            groundtruth: Tokenised tagged input (used only for shape reference).

        Returns:
            Tensor of shape ``(batch, 1, d_model)`` — the contextualised
            representation at the mask position.
        """
        token_logits = self.bert(**x).logits

        batch_size = x["input_ids"].size(0)
        batch_indices = torch.arange(batch_size, device=self.device)
        mask_token_index = (x["input_ids"] == self.tokenizer.mask_token_id).int().argmax(dim=1)

        mask_token_logits = token_logits[batch_indices, mask_token_index, :]

        # Soft embedding: differentiable replacement of the mask token
        soft_probs = torch.softmax(mask_token_logits, dim=-1)
        word_embeddings = self.bert.get_input_embeddings().weight
        predicted_embeds = torch.matmul(soft_probs, word_embeddings)

        inputs_embeds = self.bert.get_input_embeddings()(x["input_ids"])
        inputs_embeds[batch_indices, mask_token_index] = predicted_embeds.to(inputs_embeds.dtype)

        outputs = self.bert.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=x["attention_mask"],
        ).last_hidden_state

        ret = outputs[batch_indices, mask_token_index, :].unsqueeze(1)
        return ret


class Discriminator(nn.Module):
    """Cross-attention classifier that scores causal vs. non-causal.

    The query comes from the cloze (or fused) representation while the
    key/value come from the full tagged sentence encoding.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float,
        tokenizer: AutoTokenizer,
        bert: AutoModelForMaskedLM,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads)
        self.tokenizer = tokenizer
        self.bert = bert
        self.device = device

        # Feed-forward network
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.fc3 = nn.Linear(d_model, 2)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        groundtruth: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Score the event-pair representation against the tagged context.

        Args:
            x: Query tensor of shape ``(batch, 1, d_model)``.
            groundtruth: Tokenised tagged input.

        Returns:
            Logits of shape ``(batch, 1, 2)``.
        """
        key = self.bert.base_model(**groundtruth).last_hidden_state.permute(1, 0, 2)
        value = key
        x = x.permute(1, 0, 2)

        attn_output, _ = self.mha(x, key, value)
        attn_output = attn_output.permute(1, 0, 2)

        # FFN with residual connection
        out = self.dropout(self.relu(self.fc1(attn_output)))
        out = self.fc2(out)
        out = self.layer_norm(attn_output + out)
        out = self.fc3(out)

        return out


class CausalModel(nn.Module):
    """Full ECI pipeline: ClozeAnalyzer → HAN graph fusion → Discriminator.

    This model combines three sources of information:
    1. **Cloze signal** — what the LM predicts for the masked event
    2. **Syntax graph** — dependency-parsed heterogeneous graph via HANConv
    3. **Cross-attention** — discriminator attending over the tagged sentence

    Args:
        bert_path: HuggingFace model identifier (e.g. ``FacebookAI/roberta-large``).
        d_model: Hidden dimension of the transformer / HANConv.
        num_heads: Number of attention heads in the Discriminator.
        dropout_rate: Dropout probability.
        device: Target device (``'cuda'`` or ``'cpu'``).
        special_tokens: Event marker tokens to add to the tokenizer.
        visualize: If ``True``, enable verbose debug output.
    """

    def __init__(
        self,
        bert_path: str,
        d_model: int,
        num_heads: int,
        dropout_rate: float,
        device: torch.device,
        special_tokens: Optional[List[str]] = None,
        visualize: bool = False,
    ) -> None:
        super().__init__()

        # Shared BERT backbone
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        if special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.bert = AutoModelForMaskedLM.from_pretrained(bert_path)
        self.bert.resize_token_embeddings(len(self.tokenizer))

        # Sub-modules
        self.generator = ClozeAnalyzer(self.tokenizer, self.bert, device, visualize)
        self.discriminator = Discriminator(
            d_model, num_heads, dropout_rate, self.tokenizer, self.bert, device,
        )

        # Heterogeneous Graph Attention Network (HAN)
        self.device = device
        self.metadata = HAN_METADATA

        self.han = HANConv(
            in_channels=d_model,
            out_channels=d_model // 2,
            metadata=self.metadata,
            heads=2,
            dropout=dropout_rate,
        )

        # Fusion layer (BERT mask output + graph output)
        self.fusion_fc = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        groundtruth: Dict[str, torch.Tensor],
        graph_data: Dict[str, Any],
    ) -> torch.Tensor:
        """Run the full forward pass.

        Args:
            x: Tokenised masked input.
            groundtruth: Tokenised tagged input.
            graph_data: Dict with keys ``'edges'``, ``'e1_idx'``, ``'e2_idx'``
                produced by :class:`~src.data.GraphTextCollate`.

        Returns:
            Logits of shape ``(batch, 1, 2)``.
        """
        # 1. Cloze representation
        cloze_out = self.generator(x, groundtruth)

        # 2. Syntax-aware graph representation
        bert_features = self.bert.base_model(**groundtruth).last_hidden_state
        batch_size = cloze_out.size(0)

        e_graph_reps = []

        for i in range(batch_size):
            x_dict = {"word": bert_features[i]}

            # Reconstruct per-sentence graph
            edges = graph_data["edges"][i]
            if edges is None:
                edges = {}

            edge_index_dict = {}
            for rel_tuple in self.metadata[1]:
                rel = rel_tuple[1]  # e.g. 'nsubj'
                indices = edges.get(rel)

                if indices is None:
                    # Empty edge index (PyG standard: shape 2×0)
                    edge_index_dict[rel_tuple] = torch.empty(
                        (2, 0), dtype=torch.long, device=self.device,
                    )
                else:
                    edge_index_dict[rel_tuple] = torch.tensor(
                        indices, dtype=torch.long, device=self.device,
                    )

            # Forward through HAN
            han_out = self.han(x_dict, edge_index_dict)
            word_embs = han_out["word"]

            # Extract event-node embeddings
            idx_1 = min(graph_data["e1_idx"][i], word_embs.size(0) - 1)
            idx_2 = min(graph_data["e2_idx"][i], word_embs.size(0) - 1)

            e1_emb = word_embs[idx_1]  # shape: [d_model // 2]
            e2_emb = word_embs[idx_2]  # shape: [d_model // 2]

            # Concatenate event pair → relation vector
            e_pair = torch.cat([e1_emb, e2_emb], dim=-1).unsqueeze(0)  # [1, d_model]
            e_graph_reps.append(e_pair)

        e_graph_reps = torch.stack(e_graph_reps, dim=0)  # [batch, 1, d_model]

        # 3. Feature fusion (cloze + graph)
        fused_features = torch.cat([cloze_out, e_graph_reps], dim=-1)  # [batch, 1, d_model*2]
        fused_features = self.relu(self.fusion_fc(fused_features))
        fused_features = self.layer_norm(cloze_out + fused_features)

        # 4. Discriminator classification
        out = self.discriminator(fused_features, groundtruth)

        return out