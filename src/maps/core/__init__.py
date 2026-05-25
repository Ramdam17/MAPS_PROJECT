"""Core MAPS components (paper §2).

This sub-package will hold the canonical implementations of the two
load-bearing MAPS components, shared across all domains:

- ``cascade.py`` — McClelland (1989, 1979) cascade model. 50 iterations
  at integration rate α = 0.02. Paper §2.1 eq.6.
- ``second_order.py`` — Pasquali & Cleeremans (2010) comparator matrix
  + wagering head (Koch & Preuschoff 2007). Paper §2.2.
- ``losses.py`` — Contractive AutoEncoder loss (Rifai 2011), wagering
  BCE, knowledge distillation (Hinton 2015).

Empty during Sprint 10 (Phase 0/1) — populated in Sprint 11+.
"""
