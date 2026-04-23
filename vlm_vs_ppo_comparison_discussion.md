# VLM vs PPO head-to-head comparison — planning discussion

Context links:
- https://scale-icml-2026.github.io/
- https://github.com/dmasamba/vlms_multi-agent_control
- https://github.com/dmasamba/pursuit-marl-lab

---

## User

We want to achieve more to add to the paper publication for this workshop. We want a one-to-one comparison plot between VLM and PPO starting from zero-shot for VLM and cold-start for PPO and progressing to few-shots / couple of timesteps then to more shots / fully trained — basically a period-by-period comparison until they both get fully better. However we need a fair one-to-one comparison plot since they are both using different measurements.

What are your suggestions in order to achieve that?

## Assistant

The key problem is that "progress" means different things: PPO progresses via **on-policy environment steps**, VLM fine-tuning progresses via **offline training examples** (plus inference cost per action at eval time). Anything you plot has to commit to a common x-axis, or it isn't a one-to-one comparison.

### Three candidate x-axes (pick one as primary, treat others as supplements)

1. **Environment samples (recommended primary).**
   - VLM: the number of env transitions that went into the offline dataset (5k / 25k / 100k are already in these units).
   - PPO: env steps consumed during training.
   - Both are "experience seen." PPO is on-policy, VLM is expert/offline — note that asymmetry in the caption, don't hide it. This is the cleanest axis for ML readers and it directly answers "at N samples of experience, who's ahead?"

2. **Training compute (FLOPs, or GPU-hours as a proxy).**
   - Fairer when VLM inference is expensive but data-cheap, PPO data-hungry but inference-cheap. Harder to compute precisely; approximate FLOPs per gradient step × steps is fine for a workshop.
   - Good as a **second panel**, not the primary.

3. **Wall-clock training time on identical hardware.**
   - Most concrete, least theoretically pure. Useful for a practitioner-facing supplement.

### What to actually plot

- **Primary figure:** success rate (y) vs log-scale env samples (x), one curve per method: `VLM`, `MAPPO`, `IPPO`. Mean across seeds, shaded 95% CI. Mark the anchor checkpoints (zero-shot / cold-start at left, fully-trained at right). Evaluate at matched x-values where possible.
- **Secondary (Pareto / compute panel):** same y, but x = training FLOPs. Shows the story flips depending on what's scarce.
- **Tertiary (time-to-threshold bar chart):** samples required to reach e.g. 80% success on the base scenario. Optional.

### What you need to collect

You already have VLM at 5k / 25k / 100k and MAPPO/IPPO cold-start + fully-trained. To make the curve meaningful you'll need **PPO intermediate checkpoints** at roughly matched sample counts — e.g. eval MAPPO/IPPO at 5k, 25k, 100k, 1M, 10M env steps. Also useful: **VLM at 0 examples (zero-shot) already measured**, maybe a 1k or in-context "few-shot" point at the low end so the VLM curve actually has a left tail.

### Asymmetries to acknowledge in the caption, not engineer away

- PPO samples are on-policy rollouts; VLM samples are expert demonstrations. Expert data is strictly higher-quality per sample — that's part of the finding, not a bug.
- VLM inference cost is orders of magnitude higher per action. A third small panel (inference latency / cost per decision) is honest and reviewers will ask for it anyway.

### Minimal next step

Confirm which x-axis you want as primary (I'd pick env samples), and whether you can get intermediate PPO checkpoints from [pursuit-marl-lab](https://github.com/dmasamba/pursuit-marl-lab) — if yes, I can sketch the plotting script and a CSV schema for you to fill in.

---

## User

[Inspected the two fully-trained PPO runs: `PPO_2025-11-10_00-37-58` and `PPO_2025-11-11_23-49-10`.]

Here are the 2 PPO models that were fully trained. What are their total training timesteps? If we have to compare with VLM shots, how should we go about it?

Do we need different training of MAPPO?

## Assistant

### Total training budgets

| Run | Iterations | Env steps (lifetime) | Final episode return |
|---|---|---|---|
| PPO 2025-11-10 (run 1) | 5,236 | **~5.39M** | 7.75 |
| PPO 2025-11-11 (run 2) | 10,000 | **~10.29M** | (last row missing) |

So PPO consumed **5–10 million environment steps**. VLM fine-tuning maxes at 100k expert examples. That's **~50–100× more on-policy experience** than VLM's offline budget.

### The comparison problem, made concrete

You can't put them on the same "shot" axis because:
- VLM-5k = 5,000 expert transitions used in gradient updates.
- MAPPO at 5k env steps = ~1 PPO iteration, random policy, catastrophic.

A raw one-to-one plot would make PPO look absurdly bad at low x. That's not informative — it's just restating "on-policy RL is sample-inefficient."

### Recommended path

**You do not necessarily need to retrain MAPPO from scratch.** You need two things:

1. **Intermediate PPO checkpoints along its existing curve.** Right now you only have the last 10 checkpoints for each run (checkpoints 42–51 and 90–99). Those are all near-convergence, useless for a learning-curve comparison. You have two options:
   - **(a) Re-extract from `progress.csv`.** Every iteration's mean episode return is logged. You can plot PPO's training-return curve vs env steps *for free*, no re-training needed. The catch: that's training-distribution return, not your held-out success rate across shifts. Fine as a proxy for the main curve; not sufficient for shift-wise comparison.
   - **(b) Re-run PPO with denser checkpointing.** This is what I'd recommend if you want shift-wise success rates at intermediate points. Train once, save checkpoints at log-spaced env-step counts (e.g., 10k, 30k, 100k, 300k, 1M, 3M, 10M), and evaluate each on your existing shift suite. One training run, ~7 eval points per method.

2. **A second x-axis that's actually comparable.** Put both methods on **log env samples**. Mark that VLM samples are offline/expert and PPO samples are online/self-generated in the legend or caption. VLM curve is short (zero-shot, 5k, 25k, 100k) and flat-ish on the right; PPO curve is long and rises steeply. The crossover point — where PPO overtakes VLM on a given metric — *is* the headline finding.

### Concrete recommendation

- **Don't retrain MAPPO from scratch.** Instead: re-run with **denser logging/checkpointing** once. Single training run, ~1 day.
- Evaluate checkpoints at ~7 log-spaced points on your base scenario and on 2–3 key shifts.
- Plot success-rate vs env samples (log x) with VLM as ~4 scatter points (or a flat-ish line) and PPO as a rising curve.
- Caption explicitly notes: "VLM samples are expert offline transitions; PPO samples are on-policy rollouts. The axis measures experience volume, not equivalent information content."

If re-running PPO isn't feasible before the deadline, fall back to **option (a)**: use `progress.csv` training return as the PPO curve, and keep the VLM dots as your existing shift-wise success rates. It's weaker but publishable with an honest caveat.

---

## User

Alright alright I see. I'll probably re-run MAPPO with denser checkpointing. However do we also need to add 1k, 5k, 500k to the (e.g., 10k, 30k, 100k, 300k, 1M, 3M, 10M) for MAPPO?

Now if we get those denser checkpoints for MAPPO, do we need to add more few-shots for VLM as well? Cause right now we only have zero-shots, 5k, 25k, and 100k.

## Assistant

Short answer: **yes on 1k and 5k for MAPPO, no on 500k; and yes to one more VLM point on the low end.**

### MAPPO schedule — refine to

0 (cold-start), **1k, 5k**, 10k, 30k, 100k, 300k, 1M, 3M, 10M.

- **1k**: gives you a visible "still-random" anchor matching VLM zero-shot. Cheap (1 iteration), useful as a left-edge visual.
- **5k**: **this one matters** — it's the exact VLM-5k x-coordinate, so you get a literal head-to-head point on the plot, not an interpolated one.
- **500k**: skip. 300k → 1M is already close on a log axis; an extra point there doesn't add information unless the curve is weirdly non-monotonic through that range.

Keeping the schedule log-spaced with the VLM budgets (5k, 25k implicitly covered between 10k/30k, 100k) as exact points is what buys you the "same-x-axis" story.

### VLM side — add one, maybe two

Your current 4 points (zero, 5k, 25k, 100k) are enough to show the saturation curve and the headline comparison. You don't need to mirror MAPPO point-for-point; the VLM curve is expected to be short and flat-right. But two additions would meaningfully strengthen the plot:

- **VLM-1k fine-tuning** (worth doing): tells you where the jump from zero-shot to competent actually happens. Without it, the curve goes from zero-shot straight to 5k with no indication of the slope in between. Cheap relative to 100k.
- **VLM few-shot via in-context prompting (optional, nice-to-have)**: 1-shot, 3-shot, 10-shot as a *separate* series or left-tail anchor. Different mechanism from fine-tuning, so honestly label it as such — don't conflate. Only worth doing if you think in-context is part of your story; otherwise skip.

I wouldn't add more points above 100k unless you see the VLM curve still rising — and based on your existing 5k/25k/100k numbers, it's clearly saturating.

### Bottom line

- MAPPO: 10-point schedule above, **including 5k for the one-to-one anchor**.
- VLM: add **1k fine-tuned** so the left tail isn't empty. Keep the existing zero/5k/25k/100k. In-context few-shot is optional.

That gives you a defensible plot with one literal head-to-head point (at 5k env samples) plus aligned coverage everywhere else.
