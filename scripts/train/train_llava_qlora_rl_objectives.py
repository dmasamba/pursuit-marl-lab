#!/usr/bin/env python3
"""
train_llava_qlora_rl_objectives.py

VLM fine-tuning with RL-style objectives instead of pure imitation learning.

Supported training modes:
1. "imitation" (baseline): Standard behavioral cloning
2. "awr": Advantage-Weighted Regression - weight samples by advantage
3. "awac": Advantage-Weighted Actor-Critic - includes value prediction
4. "value_aux": Multi-task with auxiliary value prediction head
5. "dt": Decision Transformer style - condition on return-to-go

This enables the VLM to learn *why* actions are good, not just *what* actions to take.

Usage:
  python train_llava_qlora_rl_objectives.py \
    --data_dir distill_data_with_values \
    --val_dir val_fixed_5k \
    --model_id llava-hf/llava-v1.6-mistral-7b-hf \
    --out_dir outputs/llava_rl_awr \
    --training_mode awr \
    --advantage_clip 2.0 \
    --epochs 12 --batch_size 2 --grad_accum 8
"""

import os, json, random, argparse, copy, csv, sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from PIL import Image

from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.trainer_pt_utils import LabelSmoother

from peft import LoraConfig, get_peft_model


def parse_args():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--data_dir", type=str, required=True, help="Directory with JSONL + images (must include value/advantage fields)")
    ap.add_argument("--val_dir", type=str, default=None, help="Validation directory")
    
    # Model
    ap.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    ap.add_argument("--out_dir", type=str, default="outputs/llava_rl")
    
    # RL Training Mode
    ap.add_argument("--training_mode", type=str, default="awr",
                    choices=["imitation", "awr", "awac", "value_aux", "dt"],
                    help="Training objective: imitation (baseline), awr (advantage-weighted), "
                         "awac (actor-critic), value_aux (multi-task with value head), "
                         "dt (Decision Transformer style)")
    
    # AWR/AWAC hyperparameters
    ap.add_argument("--advantage_clip", type=float, default=2.0,
                    help="Clip advantage to [-clip, clip] before exponentiating (AWR/AWAC)")
    ap.add_argument("--awr_temperature", type=float, default=1.0,
                    help="Temperature for AWR: weight = exp(advantage / temperature)")
    ap.add_argument("--value_loss_coef", type=float, default=0.5,
                    help="Coefficient for value prediction loss (AWAC/value_aux modes)")
    ap.add_argument("--normalize_advantages", action="store_true",
                    help="Normalize advantages to mean=0, std=1 within batch")
    
    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--val_split", type=float, default=0.0)
    ap.add_argument("--dataloader_workers", type=int, default=4)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--lr_scheduler", type=str, default="cosine")
    
    # Early stopping
    ap.add_argument("--early_stop_patience", type=int, default=3,
                    help="Number of evaluations with no improvement before stopping.")
    ap.add_argument("--early_stop_min_delta", type=float, default=0.005,
                    help="Minimum absolute improvement to qualify as better.")
    ap.add_argument("--early_stop_warmup_epochs", type=int, default=3,
                    help="Don't trigger early stopping during first N epochs.")
    ap.add_argument("--early_stop_metric", type=str, default="val_loss",
                    choices=["val_loss", "val_acc"],
                    help="Metric to monitor for early stopping.")
    
    return ap.parse_args()


class VLMRLDataset(Dataset):
    """
    Dataset that loads JSONL with RL annotations (value, advantage, return_to_go, reward).
    """
    def __init__(self, root_dir: str, processor: LlavaNextProcessor, 
                 max_length: int = 1024, training_mode: str = "awr"):
        self.root = Path(root_dir)
        self.processor = processor
        self.max_length = max_length
        self.training_mode = training_mode

        self.jsonl_paths = sorted([p for p in self.root.glob("*.jsonl") if p.is_file()])
        if not self.jsonl_paths:
            raise FileNotFoundError(f"No JSONL files found in {self.root}")

        # Index all lines
        self.index = []
        for jp in self.jsonl_paths:
            with jp.open("r") as f:
                for lineno, _ in enumerate(f):
                    self.index.append((jp, lineno))
        
        print(f"[Dataset] Loaded {len(self.index)} samples from {len(self.jsonl_paths)} shards", flush=True)
        print(f"[Dataset] Training mode: {training_mode}", flush=True)

    def __len__(self):
        return len(self.index)

    def _read_line(self, jp: Path, lineno: int):
        with jp.open("r") as f:
            for i, line in enumerate(f):
                if i == lineno:
                    return json.loads(line)
        raise IndexError("Line not found")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        jp, lineno = self.index[idx]
        rec = self._read_line(jp, lineno)

        # Insert image placeholder into first user message
        messages = rec["messages"]
        msgs_with_image = copy.deepcopy(messages)
        for m in msgs_with_image:
            if m.get("role") == "user":
                contents = m.get("content", [])
                has_image = any((isinstance(c, dict) and c.get("type") == "image") for c in contents)
                if not has_image:
                    m["content"] = [{"type": "image"}] + contents
                break

        # For Decision Transformer mode, prepend return-to-go to prompt
        if self.training_mode == "dt" and "return_to_go" in rec:
            rtg = rec["return_to_go"]
            rtg_text = f"[Target return: {rtg:.2f}] "
            for m in msgs_with_image:
                if m.get("role") == "user":
                    for c in m.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "text":
                            c["text"] = rtg_text + c["text"]
                            break
                    break

        # Build texts
        full_text = self.processor.apply_chat_template(msgs_with_image, add_generation_prompt=False, tokenize=False)
        user_only = [m for m in msgs_with_image if m["role"] == "user"]
        prompt_text = self.processor.apply_chat_template(user_only, add_generation_prompt=True, tokenize=False)

        # Load image
        image = Image.open(self.root / rec["image_path"]).convert("RGB")

        # Tokenize
        enc_full = self.processor(text=full_text, images=[image], return_tensors="pt", truncation=False)
        enc_prompt = self.processor(text=prompt_text, images=[image], return_tensors="pt", truncation=False)

        input_ids = enc_full["input_ids"][0]
        attention_mask = enc_full["attention_mask"][0]
        pixel_values = enc_full["pixel_values"][0]

        image_sizes = enc_full.get("image_sizes", None)
        if image_sizes is not None:
            if isinstance(image_sizes, list):
                image_sizes = image_sizes[0]
            elif hasattr(image_sizes, "shape") and image_sizes.shape[0] == 1:
                image_sizes = image_sizes[0]

        labels = input_ids.clone()
        prompt_len = enc_prompt["input_ids"].shape[-1]
        labels[:prompt_len] = -100

        # RL annotations
        advantage = rec.get("advantage", 0.0)
        value = rec.get("value", 0.0)
        q_value = rec.get("q_value", 0.0)
        return_to_go = rec.get("return_to_go", 0.0)
        reward = rec.get("reward", 0.0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
            "image_sizes": image_sizes,
            # RL fields
            "advantage": torch.tensor(advantage, dtype=torch.float32),
            "value_target": torch.tensor(q_value, dtype=torch.float32),  # Train to predict Q(s,a)
            "return_to_go": torch.tensor(return_to_go, dtype=torch.float32),
            "reward": torch.tensor(reward, dtype=torch.float32),
        }


def _to_size_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.int64)
    return torch.tensor(x, dtype=torch.int64)


def build_collate_fn(processor: LlavaNextProcessor):
    def collate(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids = [f["input_ids"] for f in features]
        attn = [f["attention_mask"] for f in features]
        labs = [f["labels"] for f in features]
        pix = [f["pixel_values"] for f in features]
        sizes = [f.get("image_sizes") for f in features]

        batch = processor.tokenizer.pad(
            {"input_ids": ids, "attention_mask": attn, "labels": labs},
            padding=True, return_tensors="pt",
        )
        batch["pixel_values"] = torch.stack(pix, dim=0)
        if all(s is not None for s in sizes):
            batch["image_sizes"] = torch.stack([_to_size_tensor(s) for s in sizes], dim=0)
        
        # RL fields
        batch["advantage"] = torch.stack([f["advantage"] for f in features])
        batch["value_target"] = torch.stack([f["value_target"] for f in features])
        batch["return_to_go"] = torch.stack([f["return_to_go"] for f in features])
        
        return batch
    return collate


def compute_allowed_action_ids(tokenizer) -> Set[int]:
    """
    Build a set of token IDs that correspond to actions '0'..'4'.
    We include the ID of the single-token form and the last ID of the two-token form (' 0'), etc.
    """
    allowed: Set[int] = set()
    for d in list("01234"):
        ids_plain = tokenizer.encode(d, add_special_tokens=False)
        if len(ids_plain) >= 1:
            allowed.add(ids_plain[-1])
        ids_sp = tokenizer.encode(" " + d, add_special_tokens=False)
        if len(ids_sp) >= 1:
            allowed.add(ids_sp[-1])
    return allowed


@torch.no_grad()
def evaluate_loop(model, dataloader: DataLoader, tokenizer=None, training_mode: str = "awr") -> Dict[str, float]:
    """Compute mean loss and top-1 action accuracy using causal-LM shift and action-only scoring."""
    if dataloader is None:
        return {}
    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    n_batches = 0

    # Shifted accuracy
    n_correct = 0
    n_items = 0
    allowed_ids = compute_allowed_action_ids(tokenizer) if tokenizer is not None else None

    for batch in dataloader:
        # Move batch to model device
        batch = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items()}
        
        # Remove RL fields for forward pass (they're not needed for eval loss)
        batch.pop("advantage", None)
        batch.pop("value_target", None)
        batch.pop("return_to_go", None)
        
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += float(loss.detach().cpu())
        n_batches += 1

        logits = outputs.logits  # (B, T, V)
        labels = batch["labels"]  # (B, T)

        # Causal LM shift: compare logits[:, :-1] vs labels[:, 1:]
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        mask = (shift_labels != -100)
        if allowed_ids is not None:
            mask = mask & torch.isin(shift_labels, torch.tensor(list(allowed_ids), device=shift_labels.device))

        if mask.any():
            pred_ids = shift_logits.argmax(dim=-1)
            n_correct += (pred_ids[mask] == shift_labels[mask]).sum().item()
            n_items += mask.sum().item()

    if model_was_training:
        model.train()

    mean_loss = total_loss / max(1, n_batches)
    acc = (n_correct / max(1, n_items)) if n_items > 0 else 0.0
    return {"val_loss": mean_loss, "val_acc": acc, "val_count": int(n_items)}


class RLObjectiveTrainer(Trainer):
    """
    Custom Trainer that implements RL-style training objectives.
    """
    def __init__(self, training_mode: str = "awr", advantage_clip: float = 2.0,
                 awr_temperature: float = 1.0, value_loss_coef: float = 0.5,
                 normalize_advantages: bool = False, value_head: nn.Module = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.training_mode = training_mode
        self.advantage_clip = advantage_clip
        self.awr_temperature = awr_temperature
        self.value_loss_coef = value_loss_coef
        self.normalize_advantages = normalize_advantages
        self.value_head = value_head
        
        print(f"[RLTrainer] Mode: {training_mode}", flush=True)
        print(f"[RLTrainer] Advantage clip: {advantage_clip}, Temperature: {awr_temperature}", flush=True)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss based on training mode.
        """
        # Extract RL annotations
        advantages = inputs.pop("advantage", None)
        value_targets = inputs.pop("value_target", None)
        return_to_go = inputs.pop("return_to_go", None)
        
        # Forward pass
        outputs = model(**inputs)
        
        if self.training_mode == "imitation":
            # Standard cross-entropy (baseline)
            loss = outputs.loss
            
        elif self.training_mode in ["awr", "awac"]:
            # Advantage-Weighted Regression
            # Recompute per-token loss to weight by advantage
            logits = outputs.logits
            labels = inputs["labels"]
            
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Per-token cross-entropy (no reduction)
            loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())
            
            # Mask for valid tokens
            mask = (shift_labels != -100).float()
            
            # Compute per-sample loss (mean over tokens)
            per_sample_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            
            # Compute advantage weights
            if advantages is not None:
                adv = advantages.to(per_sample_loss.device)
                
                # Normalize advantages if requested
                if self.normalize_advantages and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                
                # Clip advantages
                adv = torch.clamp(adv, -self.advantage_clip, self.advantage_clip)
                
                # AWR weights: w = exp(A / temperature)
                weights = torch.exp(adv / self.awr_temperature)
                
                # Normalize weights to prevent exploding gradients
                weights = weights / weights.sum() * weights.numel()
                
                # Weighted loss
                loss = (per_sample_loss * weights).mean()
            else:
                loss = per_sample_loss.mean()
            
            # AWAC: Add value prediction loss
            if self.training_mode == "awac" and self.value_head is not None and value_targets is not None:
                # Get hidden states from last layer
                hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
                if hidden_states is not None:
                    # Use last token's hidden state for value prediction
                    last_hidden = hidden_states[:, -1, :]
                    value_pred = self.value_head(last_hidden).squeeze(-1)
                    value_loss = F.mse_loss(value_pred, value_targets.to(value_pred.device))
                    loss = loss + self.value_loss_coef * value_loss
        
        elif self.training_mode == "value_aux":
            # Multi-task: action prediction + value prediction
            loss = outputs.loss
            
            if self.value_head is not None and value_targets is not None:
                hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
                if hidden_states is not None:
                    last_hidden = hidden_states[:, -1, :]
                    value_pred = self.value_head(last_hidden).squeeze(-1)
                    value_loss = F.mse_loss(value_pred, value_targets.to(value_pred.device))
                    loss = loss + self.value_loss_coef * value_loss
        
        elif self.training_mode == "dt":
            # Decision Transformer: standard loss (return-to-go is in the prompt)
            loss = outputs.loss
        
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def make_val_split(dataset: Dataset, val_split: float, seed: int):
    n = len(dataset)
    if val_split <= 0 or n < 2:
        return dataset, None
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_split))
    return Subset(dataset, idx[n_val:]), Subset(dataset, idx[:n_val])


class MetricsLogger(TrainerCallback):
    """Collect training loss (on_log) and val metrics to save plots later."""
    def __init__(self):
        self.train_logs = []  # list of dicts with step, loss, lr, epoch
        self.val_logs = []    # list of dicts with epoch, val_loss, val_acc, n

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        rec = {"step": state.global_step, "epoch": state.epoch}
        for k in ("loss", "learning_rate", "grad_norm"):
            if k in logs:
                rec[k] = logs[k]
        self.train_logs.append(rec)

    def on_epoch_end(self, args, state, control, **kwargs):
        pass  # val handled by separate callback


class ValAtEpochEnd(TrainerCallback):
    """Run validation at epoch end and after each evaluation step."""
    def __init__(self, trainer_ref: "Trainer", val_loader: Optional[DataLoader], 
                 metrics_logger: MetricsLogger, tokenizer=None, training_mode: str = "awr"):
        self.trainer_ref = trainer_ref
        self.val_loader = val_loader
        self.metrics_logger = metrics_logger
        self.tokenizer = tokenizer
        self.training_mode = training_mode

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.val_loader is None:
            return
        metrics = evaluate_loop(self.trainer_ref.model, self.val_loader, 
                               tokenizer=self.tokenizer, training_mode=self.training_mode)
        if metrics:
            print(f"[Val] epoch={state.epoch:.2f} | val_loss={metrics['val_loss']:.4f} | "
                  f"val_acc={metrics['val_acc']:.4f} | n={metrics['val_count']}", flush=True)
            self.metrics_logger.val_logs.append({
                "epoch": float(state.epoch),
                "step": state.global_step,
                "val_loss": float(metrics["val_loss"]),
                "val_acc": float(metrics["val_acc"]),
                "n": int(metrics["val_count"]),
            })

    def on_evaluate(self, args, state, control, **kwargs):
        # Also run validation during regular eval steps
        if self.val_loader is None:
            return
        metrics = evaluate_loop(self.trainer_ref.model, self.val_loader,
                               tokenizer=self.tokenizer, training_mode=self.training_mode)
        if metrics:
            print(f"[Val] step={state.global_step} epoch={state.epoch:.2f} | "
                  f"val_loss={metrics['val_loss']:.4f} | val_acc={metrics['val_acc']:.4f} | "
                  f"n={metrics['val_count']}", flush=True)
            self.metrics_logger.val_logs.append({
                "epoch": float(state.epoch),
                "step": state.global_step,
                "val_loss": float(metrics["val_loss"]),
                "val_acc": float(metrics["val_acc"]),
                "n": int(metrics["val_count"]),
            })


class EarlyStopOnVal(TrainerCallback):
    """
    Early stop based on validation metric.
    - metric: 'val_loss' (minimize) or 'val_acc' (maximize)
    - patience: evaluations without improvement before stopping
    - min_delta: minimum absolute improvement to count as better
    - warmup_epochs: don't trigger early stopping during warmup
    """
    def __init__(self, metrics_logger: MetricsLogger, metric: str = "val_loss",
                 mode: str = "min", patience: int = 3, min_delta: float = 0.0, 
                 warmup_epochs: int = 3):
        assert metric in ("val_loss", "val_acc")
        assert mode in ("min", "max")
        self.mlog = metrics_logger
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.best = None
        self.bad_evals = 0

    def _is_improved(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return (self.best - current) > self.min_delta
        else:
            return (current - self.best) > self.min_delta

    def on_evaluate(self, args, state, control, **kwargs):
        self._check_early_stop(args, state, control)

    def on_epoch_end(self, args, state, control, **kwargs):
        self._check_early_stop(args, state, control)

    def _check_early_stop(self, args, state, control):
        # Don't trigger during warmup
        if state.epoch < self.warmup_epochs:
            return

        # Look at the last val log entry
        if not self.mlog.val_logs:
            return
        current = self.mlog.val_logs[-1].get(self.metric, None)
        if current is None:
            return

        if self._is_improved(current):
            self.best = current
            self.bad_evals = 0
            print(f"[EarlyStop] New best {self.metric}={current:.6f}", flush=True)
        else:
            self.bad_evals += 1
            print(f"[EarlyStop] No improvement on {self.metric} for {self.bad_evals}/{self.patience} eval(s).", flush=True)
            if self.bad_evals >= self.patience:
                control.should_training_stop = True
                print("[EarlyStop] Patience exhausted. Stopping training.", flush=True)


def save_metrics_and_plots(out_dir: str, m: MetricsLogger):
    """Save CSV/JSONL and simple PNG plots for training loss and validation metrics."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Save training logs
    train_csv = Path(out_dir) / "train_logs.csv"
    with train_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "loss", "learning_rate", "grad_norm"])
        w.writeheader()
        for r in m.train_logs:
            w.writerow(r)
    
    # Save validation logs
    val_csv = Path(out_dir) / "val_logs.csv"
    with val_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "val_loss", "val_acc", "n"])
        w.writeheader()
        for r in m.val_logs:
            w.writerow(r)
    
    # Plots (matplotlib)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for headless environments
        import matplotlib.pyplot as plt

        # Training loss vs steps
        steps = [r["step"] for r in m.train_logs if "loss" in r]
        tr_loss = [r["loss"] for r in m.train_logs if "loss" in r]
        if steps and tr_loss:
            plt.figure()
            plt.plot(steps, tr_loss)
            plt.xlabel("Global Step")
            plt.ylabel("Training Loss")
            plt.yscale("log")
            plt.title("Training Loss")
            plt.tight_layout()
            plt.savefig(Path(out_dir) / "training_loss.png")
            plt.close()

        # Val loss vs epoch
        epochs = [r["epoch"] for r in m.val_logs]
        val_loss = [r["val_loss"] for r in m.val_logs]
        val_acc = [r["val_acc"] for r in m.val_logs]
        if epochs:
            plt.figure()
            plt.plot(epochs, val_loss, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.title("Validation Loss")
            plt.tight_layout()
            plt.savefig(Path(out_dir) / "val_loss.png")
            plt.close()

            plt.figure()
            plt.plot(epochs, val_acc, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("Validation Accuracy")
            plt.title("Validation Accuracy")
            plt.tight_layout()
            plt.savefig(Path(out_dir) / "val_accuracy.png")
            plt.close()
        
        print(f"[Info] Plots saved to {out_dir}", flush=True)
    except Exception as e:
        print(f"[Warn] Failed to make plots: {e}", flush=True)


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir = f"{args.out_dir}_{args.training_mode}_{timestamp}"
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}", flush=True)
    print(f"VLM Training with RL Objectives", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Training mode: {args.training_mode}", flush=True)
    print(f"Data directory: {args.data_dir}", flush=True)
    print(f"Output: {args.out_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    torch.backends.cudnn.benchmark = True

    processor = LlavaNextProcessor.from_pretrained(args.model_id, use_fast=False, trust_remote_code=True)

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    # For value_aux and awac modes, we need hidden states
    output_hidden_states = args.training_mode in ["value_aux", "awac"]

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        output_hidden_states=output_hidden_states,
    )
    model.enable_input_require_grads()
    model.config.use_cache = False

    # LoRA config
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=lora_targets, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Value head for AWAC/value_aux modes
    value_head = None
    if args.training_mode in ["awac", "value_aux"]:
        hidden_size = model.config.text_config.hidden_size
        value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(model.device)
        print(f"[Info] Created value head with hidden_size={hidden_size}", flush=True)

    # Dataset
    train_dataset = VLMRLDataset(args.data_dir, processor, 
                                  max_length=args.max_length, 
                                  training_mode=args.training_mode)
    
    val_dataset = None
    if args.val_dir:
        try:
            val_dataset = VLMRLDataset(args.val_dir, processor,
                                        max_length=args.max_length,
                                        training_mode=args.training_mode)
            print(f"[Info] Validation set: {len(val_dataset)} samples", flush=True)
        except Exception as e:
            print(f"[Warn] Could not load val_dir: {e}", flush=True)
    elif args.val_split > 0:
        train_dataset, val_dataset = make_val_split(train_dataset, args.val_split, args.seed)

    collate = build_collate_fn(processor)

    # Training arguments
    targs = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        gradient_checkpointing=True,
        dataloader_num_workers=args.dataloader_workers,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    # Use custom RL trainer
    trainer = RLObjectiveTrainer(
        training_mode=args.training_mode,
        advantage_clip=args.advantage_clip,
        awr_temperature=args.awr_temperature,
        value_loss_coef=args.value_loss_coef,
        normalize_advantages=args.normalize_advantages,
        value_head=value_head,
        model=model,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate,
    )

    metrics_logger = MetricsLogger()
    trainer.add_callback(metrics_logger)

    # Build a val dataloader using the Trainer's helpers (ensures same worker/pinning config)
    val_loader = None
    if val_dataset is not None:
        val_loader = trainer.get_eval_dataloader(val_dataset)

    # Register validation callback (with tokenizer for robust accuracy)
    trainer.add_callback(ValAtEpochEnd(
        trainer_ref=trainer, 
        val_loader=val_loader, 
        metrics_logger=metrics_logger, 
        tokenizer=processor.tokenizer,
        training_mode=args.training_mode
    ))

    # Register early stopping
    if val_loader is None:
        print("[EarlyStop] Disabled: no validation set available.", flush=True)
    else:
        early_stop_mode = "min" if args.early_stop_metric == "val_loss" else "max"
        trainer.add_callback(EarlyStopOnVal(
            metrics_logger=metrics_logger,
            metric=args.early_stop_metric,
            mode=early_stop_mode,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            warmup_epochs=args.early_stop_warmup_epochs,
        ))
        print(f"[EarlyStop] Enabled: metric={args.early_stop_metric}, patience={args.early_stop_patience}, "
              f"warmup={args.early_stop_warmup_epochs} epochs", flush=True)

    # Train
    print("\n[Training] Starting...", flush=True)
    trainer.train()

    # Save
    out_adapter = os.path.join(args.out_dir, "adapter")
    trainer.model.save_pretrained(out_adapter)
    processor.save_pretrained(os.path.join(args.out_dir, "processor"))
    
    if value_head is not None:
        torch.save(value_head.state_dict(), os.path.join(args.out_dir, "value_head.pt"))

    save_metrics_and_plots(args.out_dir, metrics_logger)

    # Save training config
    config = vars(args)
    with open(os.path.join(args.out_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[Done] Training complete!", flush=True)
    print(f"  Adapter: {out_adapter}", flush=True)
    print(f"  Mode: {args.training_mode}", flush=True)


if __name__ == "__main__":
    main()
