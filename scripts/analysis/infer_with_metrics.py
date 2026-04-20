#!/usr/bin/env python3
"""
infer_with_metrics.py

Enhanced inference script with comprehensive metrics for publication.
Tracks: success rate, steps to success, episode reward, invalid actions, 
evader-seeking behavior, and validation accuracy.

Usage:
  python infer_with_metrics.py \
      --adapter_dir outputs/llava_mistral7b_25k/adapter \
      --processor_dir outputs/llava_mistral7b_25k/processor \
      --episodes 100 --seed 42 --bf16 --out_dir ./results/eval_25k
"""

import os
import re
import csv
import time
import argparse
from typing import List, Tuple, Dict, Any
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from peft import PeftModel

from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from pettingzoo.sisl import pursuit_v4
from torch.utils.tensorboard import SummaryWriter
import imageio.v2 as imageio


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================================
# METRIC COMPUTATION HELPERS
# ============================================================================

def evader_visible(obs: np.ndarray) -> bool:
    """Check if evader (red) is visible in observation."""
    evader_layer = obs[:, :, 2]  # Evader channel
    return (evader_layer > 0).any()


def get_evader_position(obs: np.ndarray) -> Tuple[int, int]:
    """Get position of evader in local observation (center of mass if multiple cells)."""
    evader_layer = obs[:, :, 2]
    positions = np.argwhere(evader_layer > 0)
    if len(positions) == 0:
        return None, None
    # Return centroid if multiple cells
    center = positions.mean(axis=0)
    return int(center[0]), int(center[1])


def moves_toward_evader(obs: np.ndarray, action: int) -> bool:
    """Check if action moves agent closer to evader (Manhattan distance)."""
    h, w = obs.shape[:2]
    agent_i, agent_j = h // 2, w // 2
    evader_i, evader_j = get_evader_position(obs)
    
    if evader_i is None:
        return False
    
    # Current distance
    curr_dist = abs(agent_i - evader_i) + abs(agent_j - evader_j)
    
    # Next position based on action
    next_i, next_j = agent_i, agent_j
    if action == 0:   # left
        next_j -= 1
    elif action == 1: # right
        next_j += 1
    elif action == 2: # down
        next_i += 1
    elif action == 3: # up
        next_i -= 1
    # action 4 (stay) doesn't change position
    
    next_dist = abs(next_i - evader_i) + abs(next_j - evader_j)
    return next_dist < curr_dist


def is_invalid_action(obs: np.ndarray, action: int) -> bool:
    """Check if action would hit wall or go out of bounds."""
    h, w = obs.shape[:2]
    agent_i, agent_j = h // 2, w // 2
    
    next_i, next_j = agent_i, agent_j
    if action == 0:   # left
        next_j -= 1
    elif action == 1: # right
        next_j += 1
    elif action == 2: # down
        next_i += 1
    elif action == 3: # up
        next_i -= 1
    elif action == 4: # stay
        return False  # stay is always valid
    
    # Check out of bounds
    if next_i < 0 or next_i >= h or next_j < 0 or next_j >= w:
        return True
    
    # Check wall collision
    if obs[next_i, next_j, 0] == 1:  # wall channel
        return True
    
    return False


def get_initial_evader_distance(env) -> float:
    """Get initial distance from pursuer to evader (for categorization)."""
    # This is environment-specific; you may need to adapt
    # For PettingZoo pursuit, we can extract from first observation
    obs = env.observe(env.agents[0])
    if isinstance(obs, dict):
        obs = obs["observation"]
    
    h, w = obs.shape[:2]
    agent_i, agent_j = h // 2, w // 2
    evader_i, evader_j = get_evader_position(obs)
    
    if evader_i is None:
        return -1
    
    return abs(agent_i - evader_i) + abs(agent_j - evader_j)


# ============================================================================
# OBSERVATION RENDERING
# ============================================================================

def build_obs_image(observation: np.ndarray, scale: int = 32) -> Image.Image:
    """Convert local observation into a color image with numeric overlays."""
    obs = observation
    h, w = obs.shape[:2]
    color_img = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background

    wall_mask = obs[:, :, 0] == 1
    color_img[wall_mask] = [0, 0, 0]

    pursuer_layer = obs[:, :, 1]
    evader_layer  = obs[:, :, 2]

    pursuer_mask = pursuer_layer > 0
    evader_mask  = evader_layer  > 0
    color_img[pursuer_mask] = [0, 200, 0]     # green
    color_img[evader_mask]  = [200, 0, 0]     # red

    # self (center) in blue
    ci, cj = h // 2, w // 2
    color_img[ci, cj] = [0, 0, 255]

    # upscale
    vis_img = np.kron(color_img, np.ones((scale, scale, 1), dtype=np.uint8))
    image = Image.fromarray(vis_img)

    # numbers
    try:
        font_size = max(10, int(scale * 0.6))
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)
    pursuer_counts = np.rint(pursuer_layer).astype(int)
    evader_counts  = np.rint(evader_layer).astype(int)

    for i in range(h):
        for j in range(w):
            cell_x = j * scale
            cell_y = i * scale

            p_count = int(pursuer_counts[i, j])
            e_count = int(evader_counts[i, j])

            if p_count > 0:
                text = str(p_count)
                margin = max(2, scale // 8)
                tx = cell_x + margin
                ty = cell_y + scale - margin - 10
                ow = 1
                for dx in [-ow, 0, ow]:
                    for dy in [-ow, 0, ow]:
                        if dx or dy:
                            draw.text((tx + dx, ty + dy), text, fill=(0, 0, 0), font=font)
                draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

            if e_count > 0:
                text = str(e_count)
                bbox = draw.textbbox((0, 0), text, font=font)
                wtxt = bbox[2] - bbox[0]
                margin = max(2, scale // 8)
                tx = cell_x + scale - wtxt - margin
                ty = cell_y + margin
                ow = 1
                for dx in [-ow, 0, ow]:
                    for dy in [-ow, 0, ow]:
                        if dx or dy:
                            draw.text((tx + dx, ty + dy), text, fill=(0, 0, 0), font=font)
                draw.text((tx, ty), text, fill=(255, 215, 0), font=font)

    return image


def annotate(img: Image.Image, text: str) -> Image.Image:
    """Add a small HUD text at the top-left of the image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    # text background
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    pad = 4
    draw.rectangle((0, 0, tw + pad * 2, th + pad * 2), fill=(0, 0, 0, 128))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return img


# ============================================================================
# PROMPT BUILDING
# ============================================================================

def build_prompt(agent_name: str, n_catch: int = 2, n_evaders: int = 1) -> str:
    """
    Build short prompt text for inference (adapts to environment configuration).
    
    Args:
        agent_name: Name of the current agent
        n_catch: Number of pursuers needed to catch an evader
        n_evaders: Number of evaders in the environment
    """
    # Adapt language for single vs multiple evaders
    evader_text = "evader" if n_evaders == 1 else "evaders"
    evader_article = "the" if n_evaders == 1 else "the"
    
    # Adapt catch rule for different n_catch values
    if n_catch == 1:
        catch_rule = "1) If you see red, move directly toward it. One pursuer can catch an evader alone."
    elif n_catch == 2:
        catch_rule = "1) If you see red, move directly toward it; if ally is on evader, move to the same spot to assist."
    else:
        catch_rule = f"1) If you see red, move directly toward it. Coordinate with allies - {n_catch} pursuers must overlap to catch."
    
    # Build catch count text
    catch_text = "One pursuer" if n_catch == 1 else f"{n_catch} pursuers"
    catch_verb = "needs" if n_catch == 1 else "must"
    
    return (
        f"You are pursuer `{agent_name}` in this local grid. Primary goal: catch {evader_article} red {evader_text}.\n"
        "Legend:\n"
        "- Blue = you (active)\n- Green = allies\n- Red = evader\n- Black = walls\n"
        "- White numbers = pursuer counts; Yellow numbers = evader counts\n\n"
        "Rules:\n"
        f"{catch_rule}\n"
        f"2) {catch_text} {catch_verb} overlap the evader to catch it.\n"
        "3) If no red visible, explore efficiently (avoid back-and-forth).\n\n"
        "Valid actions: 0=left, 1=right, 2=down, 3=up, 4=stay.\n"
        "What is your next action (0-4)? Reply as: 'action: <digit>'.\n"
    )


def extract_action(text: str) -> int:
    """Extract action from model response."""
    # Primary: look for explicit patterns like "action: 3", "move: 2", etc.
    m = re.search(r'\b(?:action|move|answer)\s*[:=]?\s*(\d)\b', text, flags=re.IGNORECASE)
    if m:
        a = int(m.group(1))
        if 0 <= a <= 4:
            return a
    # Fallback: scan for the last standalone digit 0..4 anywhere in the response
    alls = re.findall(r'\b(\d)\b', text)
    for tok in reversed(alls):
        v = int(tok)
        if 0 <= v <= 4:
            return v
    return -1


# ============================================================================
# METRICS AGGREGATOR
# ============================================================================

class MetricsAggregator:
    """Aggregate and compute statistics across episodes."""
    
    def __init__(self):
        self.episodes = []
    
    def add_episode(self, episode_data: Dict):
        """Add episode data."""
        self.episodes.append(episode_data)
    
    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics across all episodes."""
        if not self.episodes:
            return {}
        
        # Metric 1: Success Rate
        successes = [ep['success'] for ep in self.episodes]
        success_rate = np.mean(successes)
        success_std = np.std(successes)
        
        # Metric 2: Steps to Success (successful episodes only)
        success_steps = [ep['steps'] for ep in self.episodes if ep['success']]
        avg_steps_success = np.mean(success_steps) if success_steps else 0
        std_steps_success = np.std(success_steps) if success_steps else 0
        
        # Metric 3: Episode Reward
        rewards = [ep['episode_reward'] for ep in self.episodes]
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        # Metric 4: Invalid Action Rate
        invalid_rates = [ep['invalid_rate'] for ep in self.episodes]
        avg_invalid_rate = np.mean(invalid_rates)
        std_invalid_rate = np.std(invalid_rates)
        
        # Metric 5: Evader-Seeking Rate
        evader_rates = [ep['evader_seeking_rate'] for ep in self.episodes if ep['evader_seeking_rate'] >= 0]
        avg_evader_seeking = np.mean(evader_rates) if evader_rates else 0
        std_evader_seeking = np.std(evader_rates) if evader_rates else 0
        
        # Additional: Performance by distance
        close_eps = [ep for ep in self.episodes if 0 <= ep.get('initial_distance', -1) < 3]
        medium_eps = [ep for ep in self.episodes if 3 <= ep.get('initial_distance', -1) < 5]
        far_eps = [ep for ep in self.episodes if ep.get('initial_distance', -1) >= 5]
        
        return {
            # Metric 1: Success Rate
            'success_rate_mean': success_rate,
            'success_rate_std': success_std,
            'num_successes': sum(successes),
            'num_episodes': len(self.episodes),
            
            # Metric 2: Steps to Success
            'avg_steps_success_mean': avg_steps_success,
            'avg_steps_success_std': std_steps_success,
            'num_success_episodes': len(success_steps),
            
            # Metric 3: Episode Reward
            'avg_episode_reward_mean': avg_reward,
            'avg_episode_reward_std': std_reward,
            
            # Metric 4: Invalid Action Rate
            'invalid_action_rate_mean': avg_invalid_rate,
            'invalid_action_rate_std': std_invalid_rate,
            
            # Metric 5: Evader-Seeking Rate
            'evader_seeking_rate_mean': avg_evader_seeking,
            'evader_seeking_rate_std': std_evader_seeking,
            
            # Robustness by distance
            'success_rate_close': np.mean([ep['success'] for ep in close_eps]) if close_eps else 0,
            'success_rate_medium': np.mean([ep['success'] for ep in medium_eps]) if medium_eps else 0,
            'success_rate_far': np.mean([ep['success'] for ep in far_eps]) if far_eps else 0,
            'num_close': len(close_eps),
            'num_medium': len(medium_eps),
            'num_far': len(far_eps),
        }
    
    def save_summary(self, path: str):
        """Save summary to file."""
        summary = self.compute_summary()
        import json
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*60}")
        print("SUMMARY METRICS")
        print('='*60)
        print(f"1. Success Rate:        {summary['success_rate_mean']:.1%} ± {summary['success_rate_std']:.1%}")
        print(f"   ({summary['num_successes']}/{summary['num_episodes']} episodes)")
        print(f"\n2. Steps to Success:    {summary['avg_steps_success_mean']:.1f} ± {summary['avg_steps_success_std']:.1f}")
        print(f"   (based on {summary['num_success_episodes']} successful episodes)")
        print(f"\n3. Episode Reward:      {summary['avg_episode_reward_mean']:.2f} ± {summary['avg_episode_reward_std']:.2f}")
        print(f"\n4. Invalid Action Rate: {summary['invalid_action_rate_mean']:.1%} ± {summary['invalid_action_rate_std']:.1%}")
        print(f"\n5. Evader-Seeking Rate: {summary['evader_seeking_rate_mean']:.1%} ± {summary['evader_seeking_rate_std']:.1%}")
        print(f"\nRobustness by Distance:")
        print(f"   Close (<3):   {summary['success_rate_close']:.1%} ({summary['num_close']} episodes)")
        print(f"   Medium (3-5): {summary['success_rate_medium']:.1%} ({summary['num_medium']} episodes)")
        print(f"   Far (≥5):     {summary['success_rate_far']:.1%} ({summary['num_far']} episodes)")
        print('='*60)
        print(f"Summary saved to {path}\n")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", default="llava-hf/llava-v1.6-mistral-7b-hf")
    ap.add_argument("--adapter_dir", required=True, help="Path to LoRA adapter or 'none' for zero-shot")
    ap.add_argument("--processor_dir", default=None, help="Optional path to saved processor")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--merge_and_unload", action="store_true", help="Merge LoRA for faster inference")
    
    # Env params
    ap.add_argument("--x_size", type=int, default=8)
    ap.add_argument("--y_size", type=int, default=8)
    ap.add_argument("--max_cycles", type=int, default=100)
    ap.add_argument("--n_pursuers", type=int, default=2)
    ap.add_argument("--n_evaders", type=int, default=1)
    ap.add_argument("--n_catch", type=int, default=2)
    ap.add_argument("--freeze_evaders", action="store_true", default=False, help="Freeze evaders")
    
    # Logging / export
    ap.add_argument("--out_dir", type=str, default="./results/pursuit_eval")
    ap.add_argument("--fps", type=int, default=4, help="GIF frames per second")
    ap.add_argument("--save_gifs", action="store_true", help="Save per-episode GIFs")
    
    return ap.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # Set random seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = f"{args.out_dir}_seed{args.seed}_{timestamp}"
    ensure_dir(out_dir)
    
    tb_dir = os.path.join(out_dir, "tb")
    ensure_dir(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    # CSV setup
    csv_path = os.path.join(out_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "episode", "success", "steps", "episode_reward", "time_sec",
        "invalid_rate", "evader_seeking_rate", "initial_distance"
    ])
    csv_writer.writeheader()

    # Suppress transformers warnings
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    print(f"[Info] Loading model with seed={args.seed}", flush=True)
    print("[Load] Processor...", flush=True)
    processor_src = args.processor_dir if args.processor_dir and os.path.isdir(args.processor_dir) else args.base_model_id
    processor = LlavaNextProcessor.from_pretrained(processor_src, use_fast=False, trust_remote_code=True)

    print("[Load] Base model...", flush=True)
    base = LlavaNextForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply adapter if not zero-shot
    if args.adapter_dir and args.adapter_dir.lower() != "none":
        print(f"[Load] Applying adapter from: {args.adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        model = model.to(dtype)
        
        if args.merge_and_unload:
            print("[Info] Merging LoRA into base weights...", flush=True)
            model = model.merge_and_unload()
    else:
        print("[Info] Running in ZERO-SHOT mode (no adapter)", flush=True)
        model = base

    model.eval()

    # Build environment
    env = pursuit_v4.env(
        x_size=args.x_size, y_size=args.y_size, max_cycles=args.max_cycles,
        n_pursuers=args.n_pursuers, n_evaders=args.n_evaders, n_catch=args.n_catch,
        freeze_evaders=args.freeze_evaders, surround=False, shared_reward=False,
    )

    MAX_MEMORY = 5
    metrics_agg = MetricsAggregator()

    print(f"\n[Info] Running {args.episodes} episodes with seed {args.seed}...\n", flush=True)

    for ep in range(args.episodes):
        ep_start = time.time()
        env.reset(seed=args.seed + ep)  # Different seed per episode for variety
        terminated = truncated = False
        history: Dict[str, List[Tuple[Image.Image, int]]] = {}
        gif_buffers: Dict[str, List[np.ndarray]] = {}
        ep_reward_dict: Dict[str, float] = {a: 0.0 for a in env.agents}
        
        # Episode-level metrics
        invalid_actions = 0
        evader_visible_steps = 0
        moved_toward_evader_count = 0
        total_actions = 0
        
        # Get initial distance (for robustness analysis)
        initial_distance = get_initial_evader_distance(env)

        print(f"=== Episode {ep+1}/{args.episodes} (initial_dist={initial_distance:.1f}) ===", flush=True)
        steps = 0
        
        while not (terminated or truncated):
            agent = env.agent_selection
            obs, reward, terminated, truncated, _ = env.last()
            if isinstance(obs, dict):
                obs = obs["observation"]

            if agent not in history:
                history[agent] = []
                gif_buffers[agent] = []

            img = build_obs_image(obs, scale=32)
            
            # Build prompt with memory images (matches universal script structure)
            memory_images = [im for (im, _) in history[agent][-MAX_MEMORY:]]
            all_images = memory_images + [img]
            
            # Build prompt text with action history
            prompt_text = ""
            if memory_images:
                prompt_text += "Previous observations and actions:\n"
                for idx, (_, past_action) in enumerate(history[agent][-MAX_MEMORY:]):
                    prompt_text += f"Step {idx+1}: You chose action {past_action}.\n"
                prompt_text += "\n"
            
            prompt_text += build_prompt(agent, n_catch=args.n_catch, n_evaders=args.n_evaders)
            
            # Create message structure with text first, then all images (matches universal script)
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt_text}
            ] + [{"type": "image", "image": img} for img in all_images]}]
            
            # Tokenize
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=input_text, images=all_images, return_tensors="pt").to(model.device)

            # Generate (optimized for speed)
            t0 = time.time()
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=32,  # Reduced from 128 for 4x speedup
                    do_sample=False,
                    use_cache=True,  # Enable KV cache for faster generation
                    pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                )
            dt = time.time() - t0

            # Decode
            input_len = inputs["input_ids"].shape[-1]
            text = processor.decode(out[0][input_len:], skip_special_tokens=True)
            action = extract_action(text)
            
            if action == -1:
                # Fallback: try to find last valid digit in response
                last_number_match = re.findall(r'\b(\d)\b', text)
                if last_number_match:
                    valid_numbers = [int(num) for num in last_number_match if int(num) in range(0, 5)]
                    if valid_numbers:
                        action = valid_numbers[-1]  # Use the last valid number
                        print(f"[{agent}] Fallback to last valid digit -> {action}", flush=True)
                    else:
                        action = env.action_space(agent).sample()
                        print(f"[{agent}] Fallback random -> {action}", flush=True)
                else:
                    action = env.action_space(agent).sample()
                    print(f"[{agent}] Fallback random -> {action}", flush=True)
            
            # ============ METRIC TRACKING ============
            total_actions += 1
            
            # Metric 4: Invalid Action Rate
            if is_invalid_action(obs, action):
                invalid_actions += 1
            
            # Metric 5: Evader-Seeking Rate
            if evader_visible(obs):
                evader_visible_steps += 1
                if moves_toward_evader(obs, action):
                    moved_toward_evader_count += 1
            # =========================================

            # Save frame for GIF
            if args.save_gifs:
                frame = annotate(img.copy(), f"agent: {agent} | a={action}")
                gif_buffers[agent].append(np.asarray(frame))

            # Step environment
            if not (terminated or truncated):
                env.step(action)
                _, reward, _, _, _ = env.last()
                ep_reward_dict[agent] += float(reward)
                history[agent].append((img.copy(), action))
                if len(history[agent]) > MAX_MEMORY:
                    history[agent].pop(0)
            else:
                env.step(None)

            steps += 1

        # Episode ended
        ep_time = time.time() - ep_start
        success = bool(terminated and not truncated)
        episode_reward = float(sum(ep_reward_dict.values()))
        
        # Compute episode metrics
        invalid_rate = invalid_actions / max(1, total_actions)
        evader_seeking_rate = moved_toward_evader_count / max(1, evader_visible_steps) if evader_visible_steps > 0 else -1

        print(f"Episode {ep+1} | success={int(success)} | steps={steps} | "
              f"reward={episode_reward:.2f} | invalid={invalid_rate:.1%} | "
              f"evader_seeking={evader_seeking_rate:.1%} | {ep_time:.1f}s", flush=True)

        # Write CSV
        csv_writer.writerow({
            "episode": ep + 1,
            "success": int(success),
            "steps": steps,
            "episode_reward": f"{episode_reward:.6f}",
            "time_sec": f"{ep_time:.3f}",
            "invalid_rate": f"{invalid_rate:.6f}",
            "evader_seeking_rate": f"{evader_seeking_rate:.6f}",
            "initial_distance": f"{initial_distance:.2f}",
        })
        csv_file.flush()

        # TensorBoard
        writer.add_scalar("episode/success", int(success), global_step=ep + 1)
        writer.add_scalar("episode/steps", steps, global_step=ep + 1)
        writer.add_scalar("episode/reward", episode_reward, global_step=ep + 1)
        writer.add_scalar("episode/invalid_rate", invalid_rate, global_step=ep + 1)
        if evader_seeking_rate >= 0:
            writer.add_scalar("episode/evader_seeking_rate", evader_seeking_rate, global_step=ep + 1)

        # Add to aggregator
        metrics_agg.add_episode({
            'success': success,
            'steps': steps,
            'episode_reward': episode_reward,
            'invalid_rate': invalid_rate,
            'evader_seeking_rate': evader_seeking_rate,
            'initial_distance': initial_distance,
        })

        # Export GIFs
        if args.save_gifs:
            ep_dir = os.path.join(out_dir, f"ep_{ep+1:03d}")
            ensure_dir(ep_dir)
            for agent, frames in gif_buffers.items():
                if len(frames) > 0:
                    gif_path = os.path.join(ep_dir, f"{agent}.gif")
                    duration = 1.0 / max(1, args.fps)
                    imageio.mimsave(gif_path, frames, duration=duration)

    # Save summary
    summary_path = os.path.join(out_dir, "summary.json")
    metrics_agg.save_summary(summary_path)
    
    writer.flush()
    writer.close()
    csv_file.close()

    print("\nOutputs:")
    print(f"- Detailed CSV: {csv_path}")
    print(f"- Summary JSON: {summary_path}")
    print(f"- TensorBoard:  {tb_dir}")
    if args.save_gifs:
        print(f"- GIFs:         {out_dir}/ep_XXX/")
    print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    finally:
        print("\nCleaning up GPU memory...")
        torch.cuda.empty_cache()
        print("GPU cleanup complete.")
