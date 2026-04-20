"""Parity tests for SARL+CL (Sprint-08 D.22).

Mirrors the structure of ``tests/parity/sarl/``:

* ``_reference_sarl_cl.py`` — paper-faithful reference transcription of
  ``SARL_CL/examples_cl/maps.py`` (Vargas et al., MAPS TMLR) with the
  post-Sprint-08 paper Table 11 constants and paper-faithful fixes
  (GAMMA=0.999, step_size2=0.0002, etc.) already baked in.
* ``test_tier1_forward.py`` — forward-pass equivalence for SarlCLQNetwork,
  SarlCLSecondOrderNetwork, and AdaptiveQNetwork against the reference.
* ``test_tier3_update.py`` — one-step update parity for
  ``sarl_cl_update_step`` vs ``reference_dqn_update_step_cl`` on the
  non-CL degenerate branch and the 3-term CL branch.
"""
