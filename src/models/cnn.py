"""CNN classifier -- NOT YET IMPLEMENTED (Paper A, Sec. III B 3).

Spec, for when this is built (CLAUDE.md Sec. 7.3):

* TensorFlow/Keras
* four 1D convolutional layers; final dense layer, 2 neurons, softmax
* filter counts scale with lattice size: layer 1 = 2N+1, then half, quarter,
  eighth (round up)
* kernel size 3 -- the lattice size is odd, so no padding is needed
* dropout after each layer: 0.5, 0.5, then 0.2 on the final layer
* loss ``SparseCategoricalCrossentropy``, 10 epochs
* for random translation ONLY: L2 kernel regularisation lambda = 0.01 on every
  layer, needed to control overfitting on the more varied distributions
* retrained per lattice size (filter counts change); network *structure* fixed
"""
