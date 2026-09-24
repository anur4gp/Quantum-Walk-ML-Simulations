# Progress log

Engineering log: what was built, what worked, what didn't, what broke on the
way, and what is still open. The *scientific* record — definitions, methods,
numbers, tables — lives in [`project_reference.tex`](project_reference.tex).
Keep them separate: this file is for the reasoning and the dead ends, that
one is for things that end up in the thesis.

Newest session at the top.

---

## 2026-09-22 — comment diet across the codebase

No behaviour changed. Docstrings and comments were cut back to one or two
lines: what a function computes and the paper section it comes from, plus a
comment only where the code would otherwise be misread. Docstring lines across
`src/`, `scripts/` and `tests/` went 941 -> 563; comment lines 246 -> 220.

The reasoning that was living in module docstrings already exists in
`CLAUDE.md`, `project_reference.tex` and this log, so it was duplicated in
three places and drifting. Kept in the code: paper-section citations, the
class-order check in both `predict_proba`s, the P_max mode note in
`preprocess.py`, the `usable_from` gate rationale, and the placeholder-window
warnings in both training scripts.

Also: printed caveat paragraphs in the scripts were shortened to one or two
lines each (same warnings, fewer words); `_wrap` in `train_mlp.py` was dropped
for a direct `textwrap.wrap` call; the unused `evolve` import was removed from
`tests/test_walk.py`; README's test count is now 172 and the run list includes
`train_mlp_evolution.py`.

Suite still **172 passed**, and `train_mlp.py --regime quick`,
`train_svm.py`, `train_mlp_evolution.py --quick` and `legacy_classical_walk.py`
were re-run and print the same numbers as before (`MoI/N^2 = 0.293128` at
N=50, classical alpha 1.0000, quantum 1.9851).

---

## 2026-09-17 — full-evolution MLP, time-resolved critical point

Follows the `CNN test` note: *"MLP full evolution dist (see if it can catch
transition) / get dist from 3 types of walks and putting them into MLP for
initial classification / get time point for critical point."*

This is the first thing in the project that touches the **intermediate
regime** — a critical point cannot be read off the two extremes. Phase 1's
sequencing is otherwise intact: no scaling fits, no N sweep, no Table I
comparison.

### Built

| Thing | Where | State |
|---|---|---|
| Trajectory driver, `P(x,t)` for one walk | `src/qw/walk.py`, `run_walk_trajectory` | done |
| Space-time dataset + unlabelled probe sweeps | `src/data/evolution.py` | done |
| `P_max = 1` for space-time samples | `src/data/preprocess.py`, `normalize_pmax_evolution` | done |
| Sec. 7.4 rules (both, plus a robustness variant) | `src/analysis/critical.py` | done |
| Vectorised MoI over a trajectory | `src/qw/observables.py`, `moment_of_inertia_series` | done |
| Training / sweep CLI | `scripts/train_mlp_evolution.py` | done |
| Tests | `tests/test_evolution.py`, `tests/test_critical.py` | done, green |

Full suite: **172 passed** (was 117). Everything is additive — no existing
function was changed.

### What the run was

`python3 scripts/train_mlp_evolution.py --save-results notes/mlp_evolution_results.json`,
4m30s. `N=80` (161 sites, 80 steps, odd lattice), 1800 samples, 80/20 split,
sim seed 0, probe seed 1, `random_state=0`, `max_iter=400`, Paper A
`(400,200,100,50)` / `alpha=1e-3`, space-time normalisation `slice`. Probe
sweep 25 control values x 40 realisations over each channel's full range.
**Training windows are still the placeholders from `scripts/train_mlp.py`** —
every critical value below is conditional on them.

### Three quantities, deliberately kept apart

- **onset time** — first step from which the two *training windows* are
  separable. A property of the classifier.
- **critical control value** — Sec. 7.4 MLP rule (first probe point with
  `P(deloc) < 0.5`), computed at every time step so it can be watched drift.
- **crossover time `t_c`** — the same rule read along the *time* axis at fixed
  randomness. This is the "time point for the critical point": localization
  needs time to set in, so a walk in the transition region reads delocalized
  early and localized late.

### Results

| Channel | full evol. test | snapshot test | onset t | critical at t=N | `t_c` range | MoI `t_c` range |
|---|---|---|---|---|---|---|
| `discrete_coin` | 1.0000 | 1.0000 | 2 | Δθ = 0.1527 | 2–52 | 6–69 |
| `continuous_coin` | 1.0000 | 1.0000 | 2 | Δθ_M = 0.2618 | 2–79 | 17–72 |
| `random_translation` | 1.0000 | 1.0000 | 7 | P_r = 0.1250 | 7–79 | 3–74 |

`t_c` is the sustained variant (see below). The MoI column is the manual
comparator of CLAUDE.md Sec. 2.5 computed on the *same* probe walks — a
different estimator, not a second opinion from the same method.

### What worked

- **The trajectory is the same walk.** `run_walk_trajectory`'s final slice is
  *bitwise* `run_walk`'s output at the same seed, and the evolution dataset's
  last time slice is bitwise `make_dataset`'s feature matrix. Both are tested.
  That is what lets the snapshot baseline and the full-evolution model be
  compared on one simulation rather than two.
- **`t_c` decreases monotonically with randomness strength**, which is the
  physically expected shape and is reproduced independently by the MoI
  comparator on all three channels. That agreement is the main reason to
  believe the number at all.
- **Per-slice normalisation is the right default.** The peak of `P(x,t)` decays
  as the walker spreads, so global `P_max = 1` is set by the first few steps,
  when the walker is still nearly a delta function, and crushes the late-time
  slices — exactly where the signal is. Kept `global` and `none` as flags so
  this is checkable rather than asserted.

### What didn't work / had to be changed

- **The Sec. 7.4 rule is nonsense on an untrained model.** First attempt
  reported `t_c = 1` for *every* randomness strength including `Δθ = 0`. A
  model trained on `t=1` snapshots has almost no signal, its probe curve is
  arbitrary, and "first point below 0.5" dutifully reports the first step.
  Fixed with an explicit gate (`usable_from`): a time step is readable only
  once the model both separates the training windows **and** calls the weakest
  probe randomness delocalized, which is ground truth rather than a fit.
  Earlier steps come back NaN and are shaded out on the figures.
- **The plain rule still saturates at the gate boundary for strong
  randomness.** At large `Δθ` the reported `t_c` sits at the first readable
  step for all three channels. This is *not* a bug to filter away: at 2–7 steps
  nothing is localized, the walker is a few sites wide. The two training
  windows differ in coin angle as well as in localization, so an early snapshot
  is separable on coin-angle shape alone. **Separability is necessary for a
  reading, not sufficient for it to be about localization.** The sustained
  variant (first crossing the curve never returns from) is what gives the
  monotonic `t_c` in the table; the plain Sec. 7.4 rule is reported alongside
  it, not replaced.
- **Grid saturation, again.** Every accuracy in the table is 1.0000, including
  the full-evolution model down to `T = 20` steps of history. Same story as the
  2026-09-08 session: the extreme windows are trivially separable and this
  proves only that the pipeline works. The full evolution does **not** buy
  accuracy — it was never going to; what it buys is the time axis.

### Observations worth following up

- **The critical value drifts in opposite directions by channel.** As
  observation time grows, `discrete_coin` and `random_translation` drift
  *down* (0.28 → 0.15 and 0.19 → 0.125 of range) while `continuous_coin`
  drifts *up* (0.09 → 0.26). If that survives a seed sweep it is a real
  asymmetry between the two coin channels, and it bears on the exponent
  differences in Paper A Table I. **Not yet checked against a second seed.**
- **`random_translation` behaves differently and it shows in the time domain.**
  Its onset is 7 steps rather than 2, its `P(deloc)` vs `t` curves are
  strongly non-monotonic (at `P_r = 0.167` the curve crosses 0.5 repeatedly
  between t≈10 and t≈80), and its MLP `t_c` saturates at 7 while the MoI
  comparator keeps falling to 3–4. This is the known-hard channel of
  CLAUDE.md Sec. 3 appearing in a place Paper A did not look. It is the most
  promising thread here.
- The window-squeeze diagnostic from 2026-09-08 put the MLP's confusion band
  at `Δθ ≈ 0.23–0.25` (N=80); the probe crossing here gives `Δθ_c = 0.1527`.
  Different estimators, not contradictory — but worth understanding why they
  differ before either is quoted.

### Open / next

- [ ] **Seed sweep.** Every number above is one seed (sim 0, probe 1,
      sklearn 0). No error bars. This is the first thing to fix.
- [ ] Training windows are still placeholders. Research decision for you and
      Dr. Chien (CLAUDE.md Sec. 6, Sec. 9).
- [ ] The coin-angle contamination at early times deserves a cleaner
      treatment than a gate — e.g. training windows matched in `theta_0`.
- [ ] `--norm global` and `--norm none` not yet run; the claim that per-slice
      is better is reasoned, not measured.
- [ ] Nothing here is comparable to Paper A Table I (Paper A classifies
      final-time distributions). Do not put it in that table.
- [ ] Carried over: compile `project_reference.tex`; CNN still spec-only;
      `.npz` caching still not built (now more relevant — the probe sweeps
      re-simulate every run).

---

## 2026-09-08 — classical walk sim, MLP classifier, grid search, reference sheet

Phase 1 (CLAUDE.md §10). Nothing here touches the transition regime.

### Built

| Thing | Where | State |
|---|---|---|
| Classical random walk sim + comparison figures | `scripts/legacy_classical_walk.py` | done |
| MLP classifier, Paper A architecture | `src/models/mlp.py` | done |
| Named `GridSearchCV` grids | `src/models/mlp.py`, `PARAM_GRIDS` | done |
| Regime-driven training/tuning CLI | `scripts/train_mlp.py` | done |
| MLP test suite (24 tests) | `tests/test_mlp.py` | done, green |
| LaTeX reference sheet | `notes/project_reference.tex` | done, **not compiled** |

Full suite: **117 passed** (was 93 before this session).

### What worked

- **Classical walk as an exact binomial** rather than Monte Carlo. `P(x) =
  Binom(N, 1/2)[(x+N)/2]` on the reachable sites, zero elsewhere. No sampling
  noise, instant at any `N`, and it lands on the same odd lattice as
  `qw.walk.lattice_positions`, so the two curves are directly comparable site
  by site. The Monte Carlo path is still there behind `--mc` as a visual
  sanity overlay; at 20k walkers the dots sit on the exact curve.
- **The spreading figure is a free correctness check on the QW code.** Fitting
  `MoI ~ N^alpha` gives 1.996 (quantum) and 1.000 (classical) against the
  expected 2 and 1, and `MoI/N^2 = 0.29290` at `N=300, theta=pi/4` reproduces
  `1 - 1/sqrt(2) = 0.29289` to five digits. That is the textbook Hadamard-walk
  variance coefficient, so the evolution driver is right and not just
  plausible-looking.
- **Subclassing `BaseEstimator`/`ClassifierMixin`** on the MLP wrapper. This
  was the thing that made the PI's `GridSearchCV` request work cleanly: the
  wrapper is clonable, so the grid can tune `normalize` (the `P_max = 1` flag)
  as just another hyperparameter alongside `alpha` and the layer sizes,
  instead of needing a `Pipeline` with a `FunctionTransformer` wrapped around
  it. It still satisfies the plain `Classifier` protocol in `models/base.py` —
  checked by a test.
- **Regimes as a dict at the top of the script.** `--list` prints them,
  `--regime NAME` runs one, `--all` runs every one and prints a summary table.
  Adding a testing regime is one dict entry and nothing else.
- **Speed is a non-issue.** 1800 walks at `N=80` simulate in ~1.1 s; the
  400-200-100-50 net converges in 16–30 iterations, ~0.2 s. A 9-point grid at
  3-fold CV is 3 s. Iterating on this is essentially free, which was not the
  expectation going in.

### What didn't work / had to be changed

- **First comparison figure was unreadable.** Plotting every lattice site
  draws both curves as combs, because half the sites are unreachable by parity
  after `N` steps. The existing reference figures (`Total_probabilities.png`)
  do exactly this and it is fine for a single QW curve, but overlaying a
  classical Gaussian on top of it turned into a solid black block. Fixed by
  plotting the reachable sub-lattice by default, with `--all-sites` to get the
  old convention back. **This is a plotting choice only** — every observable
  is still computed on the full lattice.
- **Grid search on Phase-1 data measures nothing.** Every point of the
  3×3 architecture × alpha grid scores CV accuracy exactly `1.0000`, std
  `0.0000`. This is not a bug and not a surprise in hindsight — at the two
  extreme regimes the problem is one-peak vs two-peak, which everything
  solves — but it does mean *hyperparameter tuning cannot be completed in
  Phase 1*. The script now detects a saturated grid (score spread < 0.005) and
  says so in the output rather than letting a meaningless "best params" line
  stand unqualified. Real tuning has to wait for a transition-regime score to
  select on, i.e. Phase 2.
- **Worked around, not solved: needed a grid that discriminates.** Added a
  `stress` regime whose two windows are squeezed until they nearly touch, so
  the grid has something to rank. It is labelled DIAGNOSTIC ONLY in the
  script, because those windows straddle the transition and would corrupt any
  critical-value estimate. Do not let it leak into a results table.

### Problems hit along the way

- **`sklearn`'s binary MLP is not literally Paper A's network.** Paper A
  (CLAUDE.md §7.2) specifies a 2-neuron output layer.
  `sklearn.neural_network.MLPClassifier` uses a *single* logistic output unit
  for a binary problem. Equivalent up to reparametrization, and
  `predict_proba` still returns `(n, 2)` as the contract requires, so this is
  cosmetic — but it is a real difference from the paper's figure and is
  documented in the module docstring and in the reference sheet. Worth one
  sentence in the thesis methods section. The CNN, being Keras with a genuine
  2-neuron softmax, is where this can be checked if it ever matters.
- **Paper A does not pin the optimizer.** Layer sizes and `alpha` are
  specified; solver, learning rate, batch size, and iteration count are not.
  Those keep sklearn defaults and are exposed as constructor arguments. Any
  number that depends on them is *our* choice, not a paper value — flagged in
  the docstring so it does not get cited as one.
- **Repo reality vs `CLAUDE.md` §5.** Two mismatches, neither touched:
  - The venv is **Python 3.9.6**, but `CLAUDE.md` §8 and the README say 3.11+.
    Everything written this session uses `from __future__ import annotations`
    and runs on 3.9, so nothing is broken, but the stated floor is not what is
    installed.
  - `notes/` is a **new top-level directory**, not in the §5 layout. The
    reference sheet and this log had nowhere else sensible to live.
- **Could not compile the LaTeX.** No `pdflatex`/`xelatex`/`latexmk`/`tectonic`
  on this machine. `project_reference.tex` was instead checked
  programmatically: 38 balanced environments, 481 balanced braces, 22 `\ref`s
  all resolving to defined labels, no duplicate labels, 28 named blocks all
  paired, and no unescaped `_ # &` in text mode. It uses only `amsmath`,
  `amssymb`, `booktabs`, `longtable`, `geometry`, `hyperref`, `xcolor`, so it
  should build on any standard MacTeX install — but **it has not actually been
  built, so compile it once before relying on it.**

### Preliminary results

All from `scripts/train_mlp.py`, architecture `(400,200,100,50)`,
`alpha=0.001`, `P_max=1` normalization on, `random_state=0`, 80/20 split,
simulation seed 0. Windows are the placeholders in that script's `REGIMES`.

| Regime | Channel | N | Samples | Train | Test |
|---|---|---|---|---|---|
| `discrete_coin` | discrete | 80 | 1800 | 1.0000 | 1.0000 |
| `discrete_coin_narrow` | discrete | 80 | 1800 | 1.0000 | 1.0000 |
| `discrete_coin_wide` | discrete | 80 | 1800 | 1.0000 | 1.0000 |
| `continuous_coin` | continuous | 80 | 1800 | 1.0000 | 1.0000 |
| `random_translation` | random transl. | 80 | 1800 | 1.0000 | 1.0000 |
| `quick` | discrete | 40 | 400 | 1.0000 | 1.0000 |
| `large_lattice` | discrete | 300 | 600 | 1.0000 | 1.0000 |

**These numbers prove the pipeline works and nothing else.** Paper A says as
much for the SVM (§III B 1) and it applies identically here.

**Window squeeze (600 samples, seed 0, `discrete_coin`, N=80).** Test accuracy
stays at exactly 1.0000 until the two windows nearly touch:

| deloc window | loc window | test acc |
|---|---|---|
| [0, 0.20] | [0.30, θ₀] | 1.0000 |
| [0, 0.22] | [0.28, θ₀] | 0.9917 |
| [0, 0.24] | [0.26, θ₀] | 0.9667 |

So the band the MLP cannot separate sits near Δθ ≈ 0.23–0.25 at N=80.
**This is a diagnostic, not a critical value** — the confusion-point rules of
CLAUDE.md §7.4 have not been applied. Do not quote it as Δθ_c.

**Coarse grid on the `stress` regime** (the only Phase-1 setting where the
grid discriminates at all): Paper A's `(400,200,100,50)` ranks first at
CV 0.9583 ± 0.0106, against 0.9542 ± 0.0118 for `(100,50)`. Error bars
overlap almost entirely — consistent with Paper A's choice, not evidence for
it.

### Observations

- The MLP's first-layer sensitivity (per-site ‖W₁‖₂) is near zero on the
  unreachable parity sites and peaks near the ballistic edge, |x| ≈ 65 at
  N=80. The network keys on the edge of the light cone, which is the
  physically right feature. It is a *sensitivity*, not a decision boundary —
  unlike the linear SVM's weights it cannot be read directionally, and the
  plot panel says so.
- `random_translation` scoring 1.0000 is **not** the suspicious case CLAUDE.md
  §9 warns about. At p_r ≈ 0 vs p_r ≈ 0.5 the classes are genuinely easy. The
  known MLP/CNN failure is a transition-regime effect driven by the three-peak
  structure, so it cannot show up until Phase 2. Noting this so a clean
  Phase-1 number is not read as the problem being absent.

### Open / next

- [ ] **Compile `project_reference.tex` once** to confirm it builds.
- [ ] Training windows are still the placeholders inherited from
      `configs/svm_*.json`. Choosing them is a research decision for you and
      Dr. Chien (CLAUDE.md §6, §9) — every accuracy above is conditional on
      them.
- [ ] CNN (`src/models/cnn.py`) is still spec-only.
- [ ] `.npz` caching in `src/data/generate.py` still not built. Cheap at these
      sizes; will matter at N=1000.
- [ ] Sample-size sensitivity check (Paper A Fig. 4(a),(c)) not yet run.
- [ ] Re-run the grids once Phase 2 gives a transition-regime score to select
      on. Tuning against Phase-1 accuracy is not measuring anything.
