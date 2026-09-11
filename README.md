# Quantum Walks with Classical Randomness — ML Localization Transition

Computational physics project in the group of Dr. Chih-Chun Chien, Department of
Physics, UC Merced. Discrete-time quantum walks with classical randomness, with
supervised classifiers used to locate the delocalization → localization
transition automatically.

The specification is [`CLAUDE.md`](CLAUDE.md) and the two papers in
[`papers/`](papers/). Where this README and `CLAUDE.md` disagree, `CLAUDE.md`
wins.

## Layout

```
src/qw/          physics: operators, evolution, randomness channels, observables
src/qw/legacy/   pre-restructure walker.py / walker2N.py, kept as a reference
src/data/        sample generation, labelling, caching, P_max normalization
src/models/      classifiers (SVM, MLP, CNN) — share the base.Classifier protocol
src/analysis/    critical-value extraction, scaling fits   [later phases]
src/plotting/    shared matplotlib conventions
scripts/         thin CLI entry points, no physics
configs/         run configs; every figure regenerable from a config + seed
tests/           pytest suite
notes/           project reference sheet (LaTeX) and the progress log
data/            cached .npz simulations (gitignored)
figures/         publication-DPI output
papers/          Paper A (PRE 108, 035308) and Paper B (PRE 110, 064124)
```

Physics lives in `src/qw/`, ML in `src/models/`. **They talk only through
arrays** — no model imports a walk simulator.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+. The stack is pinned to Paper A's (`numpy`, `scipy`, `matplotlib`,
`scikit-learn`, `tensorflow`, `pytest`); ask before adding to it.

## Running

```bash
pytest                                                    # 117 tests
python scripts/plot_channels.py configs/pure_vs_random.json
python scripts/plot_channels.py configs/pure_vs_random.json --save my_figure

python scripts/train_svm.py configs/svm_discrete_coin.json          # train + report
python scripts/train_svm.py configs/svm_discrete_coin.json --plot    # + learned weights

python scripts/train_mlp.py                              # default regime, train + report
python scripts/train_mlp.py --list                       # available regimes and grids
python scripts/train_mlp.py --regime continuous_coin --plot
python scripts/train_mlp.py --grid coarse                # exhaustive GridSearchCV
python scripts/train_mlp.py --all                        # every regime, summary table

python scripts/legacy_classical_walk.py                  # classical vs quantum reference
```

`train_svm.py` takes `--channel/--n/--n-samples/--seed/--no-normalize` overrides
for quick exploration; put anything you intend to quote in a config instead.
`train_mlp.py` works the same way, but its testing regimes are named entries in
a `REGIMES` dict at the top of the script rather than separate config files.

`pytest` resolves `src/` via `pythonpath` in `pyproject.toml`; scripts insert it
themselves.

## Status

Phase 1 (classifiers in isolation) — see `CLAUDE.md` §10 for the checklist.

| Piece | State |
|---|---|
| `src/qw/` operators, walk, randomness, observables | done, tested |
| `src/data/preprocess.py` (`P_max = 1`) | done, tested |
| `src/plotting/style.py` | done |
| `src/models/base.py` protocol | done |
| `src/models/svm.py` | done, tested — 100% on all three channels |
| `src/models/mlp.py` | done, tested — 100% on all three channels; `GridSearchCV`-ready |
| `src/data/generate.py` | labelling done, tested; **`.npz` caching not yet built** |
| `src/models/cnn.py` | spec only, not implemented |
| `src/analysis/` | later phases |

## Conventions worth knowing before reading the code

- A state is a `(L, 2)` `complex128` array; columns are `(+, -)`.
- Lattice parity is always an explicit parameter. Paper A: odd, `2N+1` sites,
  centred on `x = 0`. Paper B: even, `2N` sites, no `x = 0`.
- Randomness is drawn up front into a `Schedule`, never inside the evolution
  loop. Every function that draws takes an explicit `rng`.
- Labels: `0 = delocalized`, `1 = localized`. Never flipped.
- `generate.py` returns samples with physical normalization (`sum P = 1`); the
  `P_max = 1` ML normalization is applied inside the classifier, so one flag
  covers both training and inference.
- Training windows have no defaults. They are a recorded result, not an
  implementation detail — the ones in `configs/svm_*.json` are placeholders.
