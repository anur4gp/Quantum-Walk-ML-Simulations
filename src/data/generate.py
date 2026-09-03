"""Sample generation, labelling and caching.

NOT YET IMPLEMENTED -- Phase 1 work item (CLAUDE.md Sec. 10).

The contract it must satisfy is already fixed (CLAUDE.md Sec. 6):

* One sample = the final-time probability distribution of one walk
  realisation, shape ``(2N+1,)`` float64 on the Paper A odd lattice.
* Metadata travelling with every sample: ``channel``, ``theta_0``, ``N``,
  ``control_value``, ``seed``.
* Labels: ``0 = delocalized`` (weak randomness), ``1 = localized`` (strong).
  Never flipped.
* Training windows over the control parameter are a recorded *result*, not an
  implementation detail -- they go in the run config.
* Default 1800 training samples, 80/20 train/test split.
* Cache to ``data/*.npz`` keyed by a hash of the config; log every cache hit
  and miss, never regenerate silently.
"""
