from typing import Any

import numpy as np
import torch

_NODE_STYLE = {
    0: {"label": "Drone", "color": "tab:blue", "marker": "^", "size": 130},
    1: {"label": "Obstacle", "color": "tab:gray", "marker": "o", "size": 90},
    2: {"label": "Target", "color": "tab:green", "marker": "*", "size": 160},
}


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

    def plot_spatial_attention(
        self, obs: dict[str, Any], drone_indices: list[int], rollout: torch.Tensor, top_k: int = 4, save_path: str | None = None
    ):
        """Overlay the attention rollout on the real drone/obstacle/target positions: an edge
        from each drone to its top_k highest-attention neighbors, thickness/opacity = weight.
        Node order matches state_processing.graph_builder.GraphBuilder (drones, obstacles, targets)."""
        import matplotlib.pyplot as plt

        positions, node_types, labels = [], [], []
        for i, (drone_id, d) in enumerate(obs["drones"].items()):
            positions.append(d["pos"])
            node_types.append(0)
            labels.append(drone_id)
        for obs_data in obs["obstacles"]:
            positions.append(obs_data["pos"])
            node_types.append(1)
            labels.append(None)
        for target_id, t in obs["targets"].items():
            positions.append(t["pos"])
            node_types.append(2)
            labels.append(target_id)

        positions = np.array(positions, dtype=np.float32)
        node_types = np.array(node_types, dtype=np.int64)
        n = len(positions)
        weights = rollout.detach().cpu().numpy() if torch.is_tensor(rollout) else np.asarray(rollout)
        weights = weights[:n, :n]

        fig, ax = plt.subplots(figsize=(8, 8))
        for i in drone_indices:
            order = np.argsort(-weights[i])
            drawn = 0
            for j in order:
                if j == i or drawn >= top_k:
                    continue
                w = float(weights[i, j])
                if w <= 0:
                    continue
                ax.plot(
                    [positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]],
                    color="crimson", alpha=min(1.0, w * 4), linewidth=1 + 4 * w, zorder=1,
                )
                drawn += 1

        for t in (1, 2, 0):
            idx = np.where(node_types == t)[0]
            if len(idx) == 0:
                continue
            style = _NODE_STYLE[t]
            ax.scatter(
                positions[idx, 0], positions[idx, 1], c=style["color"], marker=style["marker"],
                s=style["size"], label=style["label"], edgecolors="black", zorder=3,
            )

        for i, label in enumerate(labels):
            if label is not None:
                ax.annotate(label, (positions[i, 0], positions[i, 1]), textcoords="offset points", xytext=(6, 6), fontsize=8)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Spatial Attention Overlay (edge = drone's top attended neighbors)")
        ax.legend(loc="upper right")
        ax.set_aspect("equal")
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig
