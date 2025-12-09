from typing import Dict, List, Optional, Tuple

from . import default_agents
from models import ModelWrapper, _past_length
from utils import extract_gsm8k_answer, normalize_answer
import torch
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False
        self.task = getattr(args, "task", "gsm8k")

        # Dynamically import the appropriate prompt module based on --include_old_prompts flag
        include_old_prompts = bool(getattr(args, "include_old_prompts", False)) if args else False
        if include_old_prompts:
            prompts_module = importlib.import_module("prompts v2")
        else:
            prompts_module = importlib.import_module("prompts")

        self.build_agent_message_sequential = prompts_module.build_agent_message_sequential_latent_mas
        self.build_agent_message_hierarchical = prompts_module.build_agent_message_hierarchical_latent_mas

        # KNN filtering parameters
        self.knn_filter = bool(getattr(args, "knn_filter", False)) if args else False
        self.knn_percentage = getattr(args, "knn_percentage", 0.8) if args else 0.8  # Keep 80% of tokens
        self.knn_min_keep = getattr(args, "knn_min_keep", 5) if args else 5
        self.knn_strategy = getattr(args, "knn_strategy", "top") if args else "top"  # "top", "bottom", or "random"

        # Heatmap visualization parameters
        self.show_heatmaps = bool(getattr(args, "show_heatmaps", False)) if args else False
        self.show_heatmaps_singlelayer = bool(getattr(args, "show_heatmaps_singlelayer", False)) if args else False
        self.heatmap_data = []  # Store data for heatmap generation

        if self.latent_only:
            self.sequential_info_only = True

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    def _knn_filter_kv_cache(
        self,
        past_kv: Optional[Tuple],
        query_hidden: torch.Tensor,
        percentage: Optional[float] = None,
    ) -> Optional[Tuple]:
        """
        Filter KV cache to keep only a percentage of the most relevant entries based on
        similarity to the current query hidden state.

        Args:
            past_kv: The past key-value cache tuple
            query_hidden: Current hidden state to use as query [batch, hidden_dim] or [batch, seq, hidden_dim]
            percentage: Percentage of tokens to keep (0.0-1.0, uses self.knn_percentage if None)

        Returns:
            Filtered past_kv with only top percentage of relevant entries, maintaining temporal order
        """
        if past_kv is None:
            return None

        percentage = percentage if percentage is not None else self.knn_percentage
        seq_len = _past_length(past_kv)

        # Calculate k based on percentage
        k = max(int(seq_len * percentage), self.knn_min_keep)
        k = min(k, seq_len)

        # If we're keeping everything or nearly everything, no filtering needed
        if k >= seq_len:
            return past_kv

        # Convert Cache object to legacy format if needed
        was_cache_object = False
        cache_class = None
        if Cache is not None and isinstance(past_kv, Cache):
            was_cache_object = True
            cache_class = past_kv.__class__
            past_kv = past_kv.to_legacy_cache()

        # Extract keys from the middle layer for similarity computation
        # past_kv format: (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        num_layers = len(past_kv)
        middle_layer_idx = num_layers // 2
        keys = past_kv[middle_layer_idx][0]  # [batch_size, num_heads, seq_len, head_dim]

        batch_size, num_heads, seq_len, head_dim = keys.shape

        # Average keys across heads: [batch_size, seq_len, head_dim]
        keys_avg = keys.mean(dim=1)

        # Prepare query: ensure it's [batch_size, hidden_dim]
        if query_hidden.dim() == 3:  # [batch_size, seq, hidden_dim]
            query_hidden = query_hidden.mean(dim=1)  # Average across sequence

        # Project query to match key dimension if needed
        hidden_dim = query_hidden.shape[-1]
        if hidden_dim != head_dim:
            # Query is in full hidden space, keys are in head space
            # Reshape query to [batch_size, num_heads, head_dim] and average across heads
            if hidden_dim == num_heads * head_dim:
                # Hidden dim perfectly divides into heads
                query_hidden = query_hidden.view(batch_size, num_heads, head_dim).mean(dim=1)
            else:
                # Use linear projection to match dimensions
                # Simply reshape and average to get to head_dim size
                query_hidden = query_hidden.view(batch_size, -1, head_dim).mean(dim=1)

        # Normalize for cosine similarity
        keys_norm = torch.nn.functional.normalize(keys_avg, p=2, dim=-1)  # [batch_size, seq_len, head_dim]
        query_norm = torch.nn.functional.normalize(query_hidden.unsqueeze(1), p=2, dim=-1)  # [batch_size, 1, head_dim]

        # Compute similarity scores: [batch_size, seq_len]
        similarity = torch.matmul(keys_norm, query_norm.transpose(-2, -1)).squeeze(-1)

        # Always keep the most recent min_keep tokens
        min_keep = min(self.knn_min_keep, seq_len)
        k_selective = k - min_keep

        if k_selective > 0:
            # Get top-k from earlier tokens based on strategy
            early_seq_len = seq_len - min_keep
            early_similarity = similarity[:, :early_seq_len]  # [batch_size, early_seq_len]

            # Select indices based on strategy
            if self.knn_strategy == "top":
                # Keep most similar tokens (highest similarity scores)
                topk_values, topk_indices = torch.topk(early_similarity, k=min(k_selective, early_seq_len), dim=1, largest=True)
            elif self.knn_strategy == "bottom":
                # Keep least similar tokens (lowest similarity scores)
                topk_values, topk_indices = torch.topk(early_similarity, k=min(k_selective, early_seq_len), dim=1, largest=False)
            elif self.knn_strategy == "random":
                # Keep random tokens (baseline)
                k_rand = min(k_selective, early_seq_len)
                # Generate random indices for each batch element
                topk_indices = torch.stack([
                    torch.randperm(early_seq_len, device=keys.device)[:k_rand]
                    for _ in range(batch_size)
                ])
            else:
                raise ValueError(f"Invalid knn_strategy: {self.knn_strategy}. Must be 'top', 'bottom', or 'random'")

            # Sort indices to maintain temporal order
            topk_indices_sorted, _ = torch.sort(topk_indices, dim=1)

            # Combine with recent tokens
            recent_indices = torch.arange(seq_len - min_keep, seq_len, device=keys.device).unsqueeze(0).expand(batch_size, -1)
            selected_indices = torch.cat([topk_indices_sorted, recent_indices], dim=1)  # [batch_size, k]
        else:
            # Just keep the most recent k tokens
            selected_indices = torch.arange(seq_len - k, seq_len, device=keys.device).unsqueeze(0).expand(batch_size, -1)

        # Filter the KV cache for all layers
        filtered_layers = []
        for layer_kv in past_kv:
            if isinstance(layer_kv, tuple):
                # layer_kv is (keys, values)
                filtered_k = self._select_indices_from_cache(layer_kv[0], selected_indices)
                filtered_v = self._select_indices_from_cache(layer_kv[1], selected_indices)
                filtered_layers.append((filtered_k, filtered_v))
            elif torch.is_tensor(layer_kv):
                filtered_layers.append(self._select_indices_from_cache(layer_kv, selected_indices))
            else:
                filtered_layers.append(layer_kv)

        filtered_past = tuple(filtered_layers)

        # Convert back to Cache object if needed
        if was_cache_object:
            filtered_past = cache_class.from_legacy_cache(filtered_past)

        return filtered_past

    def _select_indices_from_cache(self, cache_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Select specific sequence positions from cache tensor.

        Args:
            cache_tensor: [batch_size, num_heads, seq_len, head_dim]
            indices: [batch_size, k] indices to select

        Returns:
            Selected tensor: [batch_size, num_heads, k, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = cache_tensor.shape
        k = indices.shape[1]

        # Expand indices for gathering: [batch_size, num_heads, k, head_dim]
        indices_expanded = indices.unsqueeze(1).unsqueeze(-1).expand(batch_size, num_heads, k, head_dim)

        # Gather along sequence dimension
        selected = torch.gather(cache_tensor, dim=2, index=indices_expanded)

        return selected

    def _compute_all_layer_similarities(
        self,
        past_kv: Optional[Tuple],
        query_hidden: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute cosine similarities between query and all KV cache tokens for ALL layers.

        Args:
            past_kv: The past key-value cache tuple
            query_hidden: Current hidden state to use as query [batch, hidden_dim] or [batch, seq, hidden_dim]

        Returns:
            Similarity matrix: [num_layers, seq_len] containing cosine similarities
        """
        if past_kv is None:
            return np.array([])

        # Convert Cache object to legacy format if needed
        if Cache is not None and isinstance(past_kv, Cache):
            past_kv = past_kv.to_legacy_cache()

        num_layers = len(past_kv)
        seq_len = _past_length(past_kv)

        # Prepare query: ensure it's [batch_size, hidden_dim]
        if query_hidden.dim() == 3:  # [batch_size, seq, hidden_dim]
            query_hidden = query_hidden.mean(dim=1)  # Average across sequence

        # Store similarities for all layers
        all_similarities = []

        for layer_idx in range(num_layers):
            keys = past_kv[layer_idx][0]  # [batch_size, num_heads, seq_len, head_dim]
            batch_size, num_heads, seq_len, head_dim = keys.shape

            # Average keys across heads: [batch_size, seq_len, head_dim]
            keys_avg = keys.mean(dim=1)

            # Project query to match key dimension if needed
            hidden_dim = query_hidden.shape[-1]
            query_for_layer = query_hidden
            if hidden_dim != head_dim:
                if hidden_dim == num_heads * head_dim:
                    query_for_layer = query_hidden.view(batch_size, num_heads, head_dim).mean(dim=1)
                else:
                    query_for_layer = query_hidden.view(batch_size, -1, head_dim).mean(dim=1)

            # Normalize for cosine similarity
            keys_norm = torch.nn.functional.normalize(keys_avg, p=2, dim=-1)  # [batch_size, seq_len, head_dim]
            query_norm = torch.nn.functional.normalize(query_for_layer.unsqueeze(1), p=2, dim=-1)  # [batch_size, 1, head_dim]

            # Compute similarity scores: [batch_size, seq_len]
            similarity = torch.matmul(keys_norm, query_norm.transpose(-2, -1)).squeeze(-1)

            # Take first batch element (assuming batch_size=1 for visualization)
            # Convert to float32 first to avoid BFloat16 numpy conversion issues
            similarity_np = similarity[0].to(torch.float32).cpu().numpy()
            all_similarities.append(similarity_np)

        # Stack into [num_layers, seq_len]
        return np.array(all_similarities)

    def _save_heatmap(
        self,
        similarity_matrix: np.ndarray,
        transition_name: str,
        problem_idx: int,
    ):
        """
        Generate and save a heatmap of cosine similarities across all layers.

        Args:
            similarity_matrix: [num_layers, seq_len] array of similarities
            transition_name: Name of the agent transition (e.g., "planner_to_critic")
            problem_idx: Problem number for filename
        """
        if similarity_matrix.size == 0:
            return

        # Create charts directory if it doesn't exist
        os.makedirs("charts", exist_ok=True)

        num_layers, seq_len = similarity_matrix.shape

        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, seq_len * 0.15), max(8, num_layers * 0.3)))

        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            cmap="RdYlGn",
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Cosine Similarity'},
            ax=ax,
            xticklabels=max(1, seq_len // 20),  # Show every 20th token label
            yticklabels=True,
        )

        # Set labels
        ax.set_xlabel("KV Cache Token Position", fontsize=12)
        ax.set_ylabel("Layer Index", fontsize=12)
        ax.set_title(
            f"Cosine Similarity Heatmap: {transition_name.replace('_', ' → ').title()}\n"
            f"Problem #{problem_idx} | Layers: {num_layers} | Tokens: {seq_len}",
            fontsize=14,
            pad=20
        )

        # Adjust layout
        plt.tight_layout()

        # Save figure
        filename = f"charts/heatmap_{transition_name}_problem{problem_idx}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[Heatmap saved: {filename}]")

    def _compute_middle_layer_similarity(
        self,
        past_kv: Optional[Tuple],
        query_hidden: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute cosine similarities between query and KV cache tokens for the MIDDLE layer only.

        Args:
            past_kv: The past key-value cache tuple
            query_hidden: Current hidden state to use as query [batch, hidden_dim] or [batch, seq, hidden_dim]

        Returns:
            Similarity vector: [seq_len] containing cosine similarities for middle layer
        """
        if past_kv is None:
            return np.array([])

        # Convert Cache object to legacy format if needed
        if Cache is not None and isinstance(past_kv, Cache):
            past_kv = past_kv.to_legacy_cache()

        num_layers = len(past_kv)
        middle_layer_idx = num_layers // 2

        # Extract keys from the middle layer
        keys = past_kv[middle_layer_idx][0]  # [batch_size, num_heads, seq_len, head_dim]
        batch_size, num_heads, seq_len, head_dim = keys.shape

        # Average keys across heads: [batch_size, seq_len, head_dim]
        keys_avg = keys.mean(dim=1)

        # Prepare query: ensure it's [batch_size, hidden_dim]
        if query_hidden.dim() == 3:  # [batch_size, seq, hidden_dim]
            query_hidden = query_hidden.mean(dim=1)  # Average across sequence

        # Project query to match key dimension if needed
        hidden_dim = query_hidden.shape[-1]
        query_for_layer = query_hidden
        if hidden_dim != head_dim:
            if hidden_dim == num_heads * head_dim:
                query_for_layer = query_hidden.view(batch_size, num_heads, head_dim).mean(dim=1)
            else:
                query_for_layer = query_hidden.view(batch_size, -1, head_dim).mean(dim=1)

        # Normalize for cosine similarity
        keys_norm = torch.nn.functional.normalize(keys_avg, p=2, dim=-1)  # [batch_size, seq_len, head_dim]
        query_norm = torch.nn.functional.normalize(query_for_layer.unsqueeze(1), p=2, dim=-1)  # [batch_size, 1, head_dim]

        # Compute similarity scores: [batch_size, seq_len]
        similarity = torch.matmul(keys_norm, query_norm.transpose(-2, -1)).squeeze(-1)

        # Take first batch element (assuming batch_size=1 for visualization)
        # Convert to float32 first to avoid BFloat16 numpy conversion issues
        similarity_np = similarity[0].to(torch.float32).cpu().numpy()

        return similarity_np

    def _save_heatmap_singlelayer(
        self,
        similarity_vector: np.ndarray,
        transition_name: str,
        problem_idx: int,
    ):
        """
        Generate and save a single-layer heatmap with min-max normalization.

        Args:
            similarity_vector: [seq_len] array of similarities
            transition_name: Name of the agent transition (e.g., "planner_to_critic")
            problem_idx: Problem number for filename
        """
        if similarity_vector.size == 0:
            return

        # Create charts directory if it doesn't exist
        os.makedirs("charts", exist_ok=True)

        seq_len = similarity_vector.shape[0]

        # Min-max normalization to [0, 1] range
        sim_min = similarity_vector.min()
        sim_max = similarity_vector.max()
        if sim_max - sim_min > 1e-8:  # Avoid division by zero
            normalized = (similarity_vector - sim_min) / (sim_max - sim_min)
        else:
            normalized = np.zeros_like(similarity_vector)

        # Reshape to 2D for heatmap display [1, seq_len]
        data_2d = normalized.reshape(1, -1)

        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, seq_len * 0.15), 3))

        # Create heatmap
        sns.heatmap(
            data_2d,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Normalized Similarity'},
            ax=ax,
            xticklabels=max(1, seq_len // 20),  # Show every 20th token label
            yticklabels=["Middle Layer"],
        )

        # Set labels
        ax.set_xlabel("KV Cache Token Position", fontsize=12)
        ax.set_ylabel("", fontsize=12)
        ax.set_title(
            f"Single-Layer Similarity: {transition_name.replace('_', ' → ').title()}\n"
            f"Problem #{problem_idx} | Tokens: {seq_len} | Range: [{sim_min:.3f}, {sim_max:.3f}]",
            fontsize=14,
            pad=20
        )

        # Adjust layout
        plt.tight_layout()

        # Save figure
        filename = f"charts/heatmap_singlelayer_{transition_name}_problem{problem_idx}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[Single-layer heatmap saved: {filename}]")

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        # Track previous agent for heatmap transitions
        prev_agent_role = None

        for agent in self.agents:

            if self.args.prompt == "sequential":
                batch_messages = [
                    self.build_agent_message_sequential(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    self.build_agent_message_hierarchical(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]


            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                # Apply KNN filtering to past_kv before passing to next agent
                if self.knn_filter and past_kv is not None:
                    # Use current input embeddings as query for KNN
                    with torch.no_grad():
                        query_embeds = self.model.model.get_input_embeddings()(wrapped_ids)
                        # Average across sequence length: [batch, seq, hidden] -> [batch, hidden]
                        query_hidden = query_embeds.mean(dim=1)
                        past_kv = self._knn_filter_kv_cache(past_kv, query_hidden)

                # Generate heatmaps for specific agent transitions
                if (self.show_heatmaps or self.show_heatmaps_singlelayer) and past_kv is not None and prev_agent_role is not None:
                    # Define transitions we want to visualize
                    target_transitions = [
                        ("planner", "critic"),
                        ("critic", "refiner"),
                        ("refiner", "judger"),  # Note: judger is handled separately below
                    ]

                    current_transition = (prev_agent_role, agent.role)
                    if current_transition in target_transitions:
                        with torch.no_grad():
                            query_embeds = self.model.model.get_input_embeddings()(wrapped_ids)
                            query_hidden = query_embeds.mean(dim=1)

                            # Generate transition name and problem index
                            transition_name = f"{prev_agent_role}_to_{agent.role}"
                            problem_idx = len(agent_traces[0]) + 1

                            # Multi-layer heatmap
                            if self.show_heatmaps:
                                # Compute similarities for all layers
                                similarity_matrix = self._compute_all_layer_similarities(past_kv, query_hidden)
                                # Save heatmap
                                self._save_heatmap(similarity_matrix, transition_name, problem_idx)

                            # Single-layer heatmap
                            if self.show_heatmaps_singlelayer:
                                # Compute similarity for middle layer only
                                similarity_vector = self._compute_middle_layer_similarity(past_kv, query_hidden)
                                # Save single-layer heatmap
                                self._save_heatmap_singlelayer(similarity_vector, transition_name, problem_idx)

                past_kv = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:

                past_for_decoding = past_kv if self.latent_steps > 0 else None

                if self.args.think:
                        judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    judger_prompts = prompts

                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(self.model.device)
                judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                # Apply KNN filtering to past_for_decoding before judger
                if self.knn_filter and past_for_decoding is not None:
                    # Use judger input embeddings as query for KNN
                    with torch.no_grad():
                        query_embeds = self.model.model.get_input_embeddings()(judger_ids)
                        # Average across sequence length: [batch, seq, hidden] -> [batch, hidden]
                        query_hidden = query_embeds.mean(dim=1)
                        past_for_decoding = self._knn_filter_kv_cache(past_for_decoding, query_hidden)

                # Generate heatmap for refiner -> judger transition
                if (self.show_heatmaps or self.show_heatmaps_singlelayer) and past_for_decoding is not None and prev_agent_role == "refiner":
                    with torch.no_grad():
                        query_embeds = self.model.model.get_input_embeddings()(judger_ids)
                        query_hidden = query_embeds.mean(dim=1)

                        # Generate transition name and problem index
                        transition_name = f"{prev_agent_role}_to_{agent.role}"
                        problem_idx = len(agent_traces[0]) + 1

                        # Multi-layer heatmap
                        if self.show_heatmaps:
                            # Compute similarities for all layers
                            similarity_matrix = self._compute_all_layer_similarities(past_for_decoding, query_hidden)
                            # Save heatmap
                            self._save_heatmap(similarity_matrix, transition_name, problem_idx)

                        # Single-layer heatmap
                        if self.show_heatmaps_singlelayer:
                            # Compute similarity for middle layer only
                            similarity_vector = self._compute_middle_layer_similarity(past_for_decoding, query_hidden)
                            # Save single-layer heatmap
                            self._save_heatmap_singlelayer(similarity_vector, transition_name, problem_idx)

                generated_batch, _ = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": judger_tokens_batch[idx],
                            "output": final_text,
                        }
                    )

            # Update previous agent role for next iteration
            prev_agent_role = agent.role

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]

            if self.task in ["gpqa", "medqa"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
            else:  # gsm8k
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
            
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
