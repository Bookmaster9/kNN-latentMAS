from typing import Dict, List

from . import default_agents
from models import ModelWrapper
# from prompts import build_agent_messages, build_agent_messages_v6, build_agent_messages_v6_text_mas
from prompts import build_agent_messages_hierarchical_text_mas, build_agent_messages_sequential_text_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
import argparse
import pdb
import numpy as np
import os
import torch

class TextMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens_each: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.model = model
        self.max_new_tokens_each = max_new_tokens_each
        self.max_new_tokens_judger = max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.args = args
        self.method_name = "text_mas"
        self.task = getattr(args, "task", "gsm8k")

        # Embedding saving parameters
        self.save_embeddings = bool(getattr(args, "save_embeddings", False)) if args else False
        self.embeddings_data = []  # Store embeddings for saving
        self.problem_counter = 0  # Track total problems processed across all batches

    def _save_embeddings_to_file(self, embeddings_dict: Dict, problem_idx: int, method_name: str):
        """
        Save embeddings to a pickle file for later analysis.

        Args:
            embeddings_dict: Dictionary containing embeddings and metadata
            problem_idx: Problem number
            method_name: Method name (text_mas)
        """
        import pickle

        # Create embeddings directory if it doesn't exist
        os.makedirs("embeddings", exist_ok=True)

        # Create filename
        filename = f"embeddings/{method_name}_problem{problem_idx}.pkl"

        # Save using pickle to preserve structure
        with open(filename, 'wb') as f:
            pickle.dump(embeddings_dict, f)

        print(f"[Embeddings saved: {filename}]")

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        contexts = ["" for _ in range(batch_size)]
        history_contexts = ["" for _ in range(batch_size)]
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        # Initialize embeddings storage for each problem in batch
        if self.save_embeddings:
            embeddings_per_problem: List[List[Dict]] = [[] for _ in range(batch_size)]

        for agent in self.agents:

            if self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_messages_hierarchical_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=self.args,
                    )
                    for idx, item in enumerate(items)
                ]
            else:
                batch_messages = [
                    build_agent_messages_sequential_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=self.args,
                    )
                    for idx, item in enumerate(items)
                ]

            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            # Save input embeddings if requested
            if self.save_embeddings:
                with torch.no_grad():
                    input_embeds = self.model.model.get_input_embeddings()(input_ids)
                    # Average across sequence for a summary representation
                    input_hidden = input_embeds.mean(dim=1)  # [batch, hidden_dim]

            generated_texts, past_kv = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens_each,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            # Save output embeddings if requested (for each problem in batch)
            if self.save_embeddings and past_kv is not None:
                with torch.no_grad():
                    # Extract final hidden state from past_kv (approximation)
                    # We'll use the input hidden as proxy since text_mas doesn't expose hidden states
                    # Convert to float32 first to avoid BFloat16 numpy conversion issues
                    for idx in range(batch_size):
                        embeddings_entry = {
                            'agent_name': agent.name,
                            'agent_role': agent.role,
                            'input_hidden': input_hidden[idx].to(torch.float32).cpu().numpy(),  # [hidden_dim]
                            'generated_text': generated_texts[idx],  # Store the actual generated text
                        }
                        embeddings_per_problem[idx].append(embeddings_entry)

            agent_name_map_for_prompt_hierarchical = {
                "Planner": "Math Agent",
                "Critic": "Science Agent",
                "Refiner": "Code Agent",
                "Judger": "Task Summrizer",
                "planner": "Math Agent",
                "critic": "Science Agent",
                "refiner": "Code Agent",
                "judger": "Task Summrizer",
            }

            for idx in range(batch_size):

                text_out = generated_texts[idx].strip()

                if self.args.prompt == "hierarchical":
                    formatted_output = f"[{agent_name_map_for_prompt_hierarchical[agent.name]}]:\n{text_out}\n\n"
                else:
                    formatted_output = f"[{agent.name}]:\n{text_out}\n\n"

                if agent.role != "judger":

                    contexts[idx] = f"{contexts[idx]}{formatted_output}"
                    history_contexts[idx] = f"{history_contexts[idx]}{formatted_output}"
                else:
                    final_texts[idx] = text_out
                mask = attention_mask[idx].bool()
                trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
                agent_traces[idx].append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "input": prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": tokens_batch[idx],
                        "output": text_out,
                    }
                )
            # import pdb; pdb.set_trace()

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
                    "context": history_contexts[idx],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )

            # Save embeddings for this problem (using global counter)
            if self.save_embeddings and embeddings_per_problem[idx]:
                self.problem_counter += 1
                embeddings_dict = {
                    'method': 'text_mas',
                    'question': item["question"],
                    'agents': embeddings_per_problem[idx],
                }
                self._save_embeddings_to_file(embeddings_dict, self.problem_counter, 'text_mas')

        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
