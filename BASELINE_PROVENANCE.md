# Baseline Provenance and Claim Boundaries

This note corrects method labels without rewriting historical action files.
Those YAMLs remain byte-for-byte unchanged so existing artifact hashes and
sidecars stay auditable. Historical action IDs are identifiers, not evidence of
an official reproduction.

## Historical Label Corrections

| Frozen file and SHA-256 | Historical ID | What actually ran | Permitted description |
|---|---|---|---|
| `freeu_structural_pilot.yaml`, `e930e918...acc8f25b` | `freeu_official_sdxl` | diffusers' constant half-channel gain with `[0.6, 0.4, 1.1, 1.2]` | historical diffusers FreeU surrogate |
| `s5_smoke.yaml`, `c64f9f94...45a3f3`; `s5_development.yaml`, `76cd773a...4d718` | `pladis_official` | independent PLADIS port on all `attn2` layers with FP32 probabilities | historical all-layer PLADIS reproduction |
| same S5 files | `gag_official` | independent implementation of GAG Eq. 12/13 on all `attn2` layers | paper-derived GAG reproduction |

The FreeU source README uses a spatially adaptive channel-mean modulation and
reports SDXL `[s1,s2,b1,b2]=[0.9,0.2,1.3,1.4]`. Its runnable demo and diffusers
use a different constant-gain operator. The PLADIS SDXL source defaults to
`[up, down]`, excludes `mid`, computes probabilities in the query dtype, and
states diffusers `0.33.1`; this repository's matched experiment environment is
`0.32.1`. GAG has no verified author code as of 2026-08-25, so `official` is
not an admissible label.

## Corrected Implementations

- FreeU `paper_adaptive_3676d36` pins `ChenyangSi/FreeU` commit
  `3676d3652a44101f9cca030c33f82756dab249d7` (MIT), the README equation,
  and its hidden-channel dispatch (`1280 -> b1/s1`, `640 -> b2/s2`).
- PLADIS `pladis_operator_port_248b9d1` pins `cubeyoung/PLADIS` commit
  `248b9d15701c08094c47dc90b4ae24afbf5cf7a9` (Apache-2.0). It is an
  operator-level port to diffusers `0.32.1`, not an end-to-end upstream
  environment reproduction.
- GAG `gag_eq13_reimplementation_2603.02531v2` pins arXiv v2 Eq. 12/13.
  `alpha=1.5` and `[up, down]` are explicitly inherited from PLADIS. CC BY 4.0
  covers the manuscript; this repository's Apache-2.0 covers the code.

Run `eval-pipeline/generation_environment.py` against the registered lock before
new control generation. Sidecars must record the implementation identifier,
source commit or paper ID, action hash, package lock hash, runtime versions,
selected GPU identity, and the observed cross-attention processor topology.
