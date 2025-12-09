import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

from tqdm import tqdm

from data import load_gsm8k, load_gpqa_diamond, load_medqa
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
import time


class TeeLogger:
    """Logger that writes to both stdout and a file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas"], required=True)
    parser.add_argument("--model_name", type=str, required=True, choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"])
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--task", choices=["gsm8k", "gpqa", "medqa"], default="gsm8k")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=1)
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--include_old_prompts", action="store_true", help="Use prompts v2.py with old prompt included in each agent")
    parser.add_argument("--seed", type=int, default=42)

    # KNN filtering arguments (for latent_mas only)
    parser.add_argument("--knn_filter", action="store_true", help="Enable KNN filtering of KV cache")
    parser.add_argument("--knn_percentage", type=float, default=0.8, help="Percentage of tokens to keep (0.0-1.0), e.g., 0.8 keeps 80%%")
    parser.add_argument("--knn_min_keep", type=int, default=5, help="Minimum number of recent tokens to always keep")
    parser.add_argument("--knn_strategy", type=str, choices=["top", "bottom", "random"], default="top",
                        help="Strategy for selecting tokens: 'top' (most similar), 'bottom' (least similar), 'random' (random selection)")

    # Visualization arguments
    parser.add_argument("--show_heatmaps", action="store_true", help="Generate and save heatmaps of cosine similarities across all layers for agent transitions")
    parser.add_argument("--show_heatmaps_singlelayer", action="store_true", help="Generate and save single-layer heatmaps (middle layer used for kNN) with min-max normalization")

    # Embedding analysis arguments
    parser.add_argument("--save_embeddings", action="store_true", help="Save embeddings from both text_mas and latent_mas for comparison analysis")

    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model_name.replace("/", "_")
    log_filename = f"logs/{model_short}_{args.prompt}_{args.method}_samples{args.max_samples}_{timestamp}.txt"

    # Set up logging to both console and file
    logger = TeeLogger(log_filename)
    sys.stdout = logger
    sys.stderr = logger

    print(f"Logging to: {log_filename}")
    print(f"Arguments: {vars(args)}\n")

    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, args=args)
    
    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []

    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split=args.split)
    elif args.task == "medqa":
        dataset_iter = load_medqa(split=args.split)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)  
        args.max_samples = len(dataset_iter)

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                args.max_samples,
            )
            batch = []
            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=args.max_samples,
        )
    progress.close()
    
    total_time = time.time() - start_time

    acc, correct = evaluate(preds)
    print(
        json.dumps(
            {
                "method": args.method,
                "model": args.model_name,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "accuracy": acc,
                "correct": correct,
                "total_time_sec": round(total_time,4),
                "time_per_sample_sec": round(total_time / args.max_samples, 4),
            },
            ensure_ascii=False,
        )
    )

    # Close logger and restore stdout/stderr
    logger.close()
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal
    print(f"\nLog saved to: {log_filename}")


if __name__ == "__main__":
    main()
