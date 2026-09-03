"""Linear SVM classifier -- NOT YET IMPLEMENTED (Paper A, Sec. III B 1).

Spec, for when this is built (CLAUDE.md Sec. 7.1):

* ``sklearn`` linear ``SGDClassifier``
* loss ``modified_huber`` -- *not* hinge, which gives hard true/false output;
  modified Huber yields calibrated binary probabilities
* Baseline expectation is near-perfect test accuracy even at small sample
  size, because the two classes are one-peak vs two-peak and trivially
  separable. High accuracy here proves only that the pipeline is not broken.
"""
