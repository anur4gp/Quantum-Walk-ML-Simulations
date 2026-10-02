# Cheat sheet — what I did this week (commit `958b949`)

## What I did, in one breath
Before this, the MLP only saw the **final** snapshot of each walk. Now I give it
the **whole movie**, the walk at every time step, and ask *when* during the walk
it goes from "spread out" to "stuck in the middle."

## Three things I built
1. **Save every time step of a walk**, not just the last one. Same walk, same seed; it just keeps more.
   → [src/qw/walk.py:208](../src/qw/walk.py#L208)
2. **A dataset of those full walks**, plus "probe" walks at in-between randomness values to test the model on.
   → [src/data/evolution.py](../src/data/evolution.py)
3. **A script that trains the MLP and reads off the transition**, using Paper A's rule: the first point where the model says "delocalized" with less than 50% confidence.
   → [scripts/train_mlp_evolution.py](../scripts/train_mlp_evolution.py) (rule in [src/analysis/critical.py](../src/analysis/critical.py))

## What I found (N = 80, one seed)
- **Accuracy is 100% on every channel.** That's expected, because the two classes (two peaks vs one peak) are easy to tell apart. It only shows the code works.
- **Stronger randomness means the walk localizes earlier.** This holds on all 3 channels, and the MoI method (by hand) shows the same trend. That's the main takeaway.
- **Random translation acts differently,** as it did in Paper A:
  - The model needs 7 steps before it can tell the classes apart. The other two channels need 2.
  - Its answer flips back and forth over time instead of settling.

Results: [figures/mlp_evolution_summary.png](../figures/mlp_evolution_summary.png) (show this one first) · numbers in [notes/mlp_evolution_results.json](mlp_evolution_results.json)

## How to read the summary figure
Four panels, same colors throughout: **red dashed = Jittered** (discrete coin), **green dotted = Uniform Jittered** (continuous coin), **purple dash-dot = Random Transform** (random translation). All at N = 80, seed 0.

Some setup first. The model is trained on two groups of walks: very weak randomness (label "delocalized") and very strong randomness (label "localized"). Then I give it **probe walks**: walks at 25 randomness values spread across the whole range, which it never saw in training. For each probe it outputs P(delocalized). **The critical point is the first place that number drops below 50%.**

The three channels use different knobs (Δθ and Δθ_M go up to θ₀ = π/6, P_r goes up to 0.5). So the axes say **"control / range"**: 0 means no randomness, 1 means the maximum. 0.3 means "30% of the way to max randomness."

**Top left — "Can the model tell the two groups apart at step t?"**
- For every time step t, I train a separate model on just that one snapshot P(x, t), then check its test accuracy.
- At t = 1 all three sit at 50% (a coin flip). After one step every walk looks the same, so there's nothing to learn.
- The two coin channels hit 100% by step 2. Random Transform climbs more slowly: it passes 95% at step 7 and reaches 100% around step 13.
- Takeaway: this is a sanity check. It tells me from which time step onward the model's answers are worth reading.

**Top right — "If I stop the walk at step t, where is the transition?"**
- Same per-step models. At each t, I run the probe walks through and record the randomness where P(delocalized) first drops below 50%.
- Read it as "critical randomness, measured at time t." The jumpiness is partly because the probe grid is coarse (steps of 1/24 ≈ 0.04) and there's only one seed.
- Jittered and Random Transform drift **down** as t grows: a longer walk needs less randomness to localize. Uniform Jittered drifts **up**. I don't have an explanation for that yet, so I shouldn't claim one.

**Bottom left — "At a fixed amount of randomness, when does the walk localize?"** (the main result)
- This flips the top-right panel. For each probe randomness value, I record the time step t_c where the model decides the walk has become localized. Here "decides" means it drops below 50% **and stays there** (the stricter rule).
- Lines are the MLP. **Faint squares are the manual MoI method** on the same walks: the step where MoI falls below half of the no-randomness (ballistic) MoI and stays there.
- All three lines slope down: **more randomness means earlier localization.** The MoI squares slope the same way, and that agreement is the main takeaway.
- If a point is missing at the left, that walk never localized within 80 steps.
- ⚠️ The flat floors (at 2 for the coin channels, at 7 for Random Transform) are **not measurements**. They are the earliest step the model is allowed to answer (from the top-left panel). So "t_c = 2" really means "localized at step 2 or earlier."
- ⚠️ MLP and MoI agree on the trend, not the numbers. The gap is biggest for Uniform Jittered, where the green squares sit far to the right of the green line.

**Bottom right — "Does giving the model the whole movie help?"**
- Instead of one snapshot, the model sees every step from 1 to T stacked together, P(x, t ≤ T). Test accuracy vs T.
- Coin channels: 100% no matter what. Random Transform: ~93% at T = 5 and 100% from T = 20.
- Takeaway: the full history is at least as easy to classify as a single snapshot. Like every accuracy number here, it shows the pipeline works, not that it has found the transition.

## Be upfront about
- **One seed only.** There are no error bars yet.
- **Training windows are placeholders.** They are not chosen yet, and every critical value depends on them.
- **This can't be compared to Paper A's Table I.** It's a different kind of input.
- **The MLP and MoI agree on the trend, not the exact numbers.** Sometimes they are far apart.

## Questions he might ask
| Question | Short answer |
|---|---|
| Why 50% accuracy at step 1? | After one step every walk looks identical: half on the left, half on the right. There's nothing to learn. |
| Why do the coin channels separate so fast (2 steps)? | The two classes use different coin angles, so the model may be spotting the angle, not localization. I need to fix this. |
| Did you change Paper A's rule? | No. I also added a stricter version ("drops below 50% and stays there") as a sanity check, and I report both. |
| How do you know the new code is right? | Its last time step matches the old code exactly, probability always sums to 1, and all 172 tests pass. |
| Did the cleanup break anything? | No. The logic is identical; only the comments changed. |
| What's next? | Run more seeds, pick real training windows (with you), then go back to the CNN. |

## Ask him
- What training windows should I use?
- Keep going with this time-based approach, or finish the CNN first?
