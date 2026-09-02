# Latent Renderer Training Package

This package contains the shared training contracts for the TPAMI extension.
It is independent of diffusers and the production SDXL pipeline so the
probability and objective math can be tested on CPU before a GPU run.

## Modules

- contracts.py: frozen action mask, coefficient bounds, and the
  Lebesgue-plus-inactive-Dirac measure.
- distributions.py: fixed-variance transformed Gaussian sampling, exact
  tanh change-of-variables log probability, and reference KL.
- rollout.py: transition records and one shared-prefix replay helper.
- objectives.py: native-transition OPD, search distillation, renderer-DPO,
  and per-decision reference-anchored RL losses.
- ledger.py: append-only reserve-before-compute query accounting. Unfinished
  reservations remain charged after a crash.
- checkpoint.py: atomic checkpoint writes and contract-hash validation.
- trainer.py: method-agnostic optimizer and EMA update loop.
- cli.py: a dependency-light contract probe for environment checks.

The production renderer remains in AttentionGuidance/latent_renderer.py.
This package must call that renderer through an adapter; it must not copy the
SDXL pipeline or remove its no_grad boundary. Method differences belong in
versioned configs and objective selection, not duplicated runners.

Run the CPU contracts with:

    python -m pytest -q tests/test_latent_renderer_training.py
    python -m latent_renderer_training.cli --slots 6
