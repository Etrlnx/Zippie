from typing import Any

import torch


class AttentionVisualizer:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def extract_attention_rollout(self, attn_weights: list[torch.Tensor]) -> torch.Tensor:
        """Attention rollout (Abnar & Zuidema 2020): average heads per layer, mix in the
        residual identity, renormalize, then compose layers. Assumes batch size 1."""
        rollout = None
        for layer_attn in attn_weights:
            a = layer_attn.mean(dim=1)[0]
            identity = torch.eye(a.shape[-1], device=a.device)
            a = 0.5 * a + 0.5 * identity
            a = a / a.sum(dim=-1, keepdim=True)
            rollout = a if rollout is None else a @ rollout
        return rollout

    def plot_attention_matrix(self, attn_matrix: torch.Tensor, node_labels: list[str] | None = None, save_path: str | None = None):
        import matplotlib.pyplot as plt

        matrix = attn_matrix.detach().cpu().numpy() if torch.is_tensor(attn_matrix) else attn_matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(matrix, cmap="viridis")
        if node_labels is not None:
            ax.set_xticks(range(len(node_labels)))
            ax.set_yticks(range(len(node_labels)))
            ax.set_xticklabels(node_labels, rotation=90, fontsize=6)
            ax.set_yticklabels(node_labels, fontsize=6)
        fig.colorbar(im, ax=ax)
        ax.set_title("Attention Rollout")
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig
