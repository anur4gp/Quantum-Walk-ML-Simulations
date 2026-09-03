"""MLP classifier -- NOT YET IMPLEMENTED (Paper A, Sec. III B 2).

Spec, for when this is built (CLAUDE.md Sec. 7.2):

* ``sklearn`` ``MLPClassifier``
* six layers total: input (auto-sized to lattice) -> hidden 400, 200, 100, 50
  -> output 2
* regularisation ``alpha = 0.001``
* hyperparameters were explored with ``GridSearchCV`` over hidden layer sizes
  and ``alpha``
* layer sizes kept fixed from N=80 to N=1000; scaling them with lattice size
  gave negligible improvement
"""
