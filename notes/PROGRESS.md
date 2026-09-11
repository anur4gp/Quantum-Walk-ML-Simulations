# Progress log

Engineering log: what was built, what worked, what didn't, what broke on the
way, and what is still open. The *scientific* record — definitions, methods,
numbers, tables — lives in [`project_reference.tex`](project_reference.tex).
Keep them separate: this file is for the reasoning and the dead ends, that
one is for things that end up in the thesis.

Newest session at the top.

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
