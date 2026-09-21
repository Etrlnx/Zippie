from typing import Any

import torch

CAVEAT = "Attention weight is a heuristic proxy, not a verified causal explanation (Jain & Wallace, 2019)."


class NeighborAnalyzer:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def rank_influential_neighbors(self, node_idx: int, rollout: torch.Tensor, node_metadata: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        weights = rollout[node_idx].detach().cpu().numpy()
        candidates = [
            {
                "node_index": i,
                "label": node_metadata[i].get("label", f"node_{i}"),
                "type": node_metadata[i].get("type", "unknown"),
                "attention_weight": float(weights[i]),
            }
            for i in range(len(weights)) if i != node_idx
        ]
        candidates.sort(key=lambda c: c["attention_weight"], reverse=True)
        return candidates[:top_k]

    def generate_decision_summary(self, drone_id: str, node_idx: int, action: str, rollout: torch.Tensor, node_metadata: list[dict[str, Any]], top_k: int = 3) -> dict[str, Any]:
        top_neighbors = self.rank_influential_neighbors(node_idx, rollout, node_metadata, top_k=top_k)
        proxy_lines = [f"{n['label']} ({n['type']}): {n['attention_weight']:.3f}" for n in top_neighbors]
        return {
            "drone_id": drone_id,
            "action": action,
            "top_neighbors": top_neighbors,
            "summary_proxy": f"{drone_id} chose '{action}', most influenced by: " + ", ".join(proxy_lines),
            "caveat": CAVEAT,
        }
