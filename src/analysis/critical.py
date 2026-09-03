"""Confusion-point extraction -- LATER PHASE (Phase 2).

The rule is asymmetric by classifier and deliberately so (CLAUDE.md Sec. 7.4):

* SVM: the point of **maximal confusion**, where the two class probabilities
  are equal.
* MLP and CNN: these jump rather than crossing over gradually, so use the
  **first point where P(delocalized) falls below 50%**.

Known artifact: the ML methods estimate from a small *range* around the
transition rather than a single point, which is the suspected cause of their
systematically lower exponents.
"""
