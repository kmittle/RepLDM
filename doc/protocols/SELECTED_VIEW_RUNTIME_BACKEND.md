# Selected-View Runtime Backend

`eval-pipeline/data_catalog/selected_runtime_backend.py` is the only runtime
accepted by the formal selected-view authorization path. It is intentionally
offline and loads assets from the absolute paths frozen in the selected-view
config.

## Model and tokenizer

The classifier, semantic-text, and image-embedding roles must bind the same
OpenAI CLIP checkpoint. The descriptor must identify an `openai/...` model and
contain exactly one local checkpoint file (`.pt`, `.pth`, `.bin`, or `.ckpt`).
The backend calls `clip.load` with that path; it never resolves a model name or
downloads weights. Both SDXL tokenizer descriptors must contain local
`vocab.json` and `merges.txt` files in one directory. Auxiliary tokenizer files
read by `CLIPTokenizer.from_pretrained` (`tokenizer.json`,
`tokenizer_config.json`, `special_tokens_map.json`, and `added_tokens.json`)
must also be included in the descriptor when present. The first configured
tokenizer is used for OpenAI CLIP text features because the current config
schema has no separate CLIP-tokenizer field; SDXL's two tokenizers share the
OpenAI BPE vocabulary.

Protected prompt indexes and semantic-gate queries use the OpenAI CLIP context
contract: the selected tokenizer is called with `max_length=77`, padded to 77,
and deterministic truncation enabled. This is required because some held-out
captions exceed 77 tokens. Candidate training rows are checked separately with
the SDXL tokenizers and truncation disabled; a candidate that would be
truncated is rejected.

## Protected indexes

Semantic-text and image indexes are `.npz` files with exactly `ids` and
`embeddings`. `ids` is a unique one-dimensional UTF-8 string array; embeddings
are finite, non-zero, row-normalized floating-point arrays with the configured
row count. The pHash index is UTF-8 JSONL with exactly `id` and a lowercase
16-hex-digit `phash` per row. Every file is rehashed before loading, and the
runtime reports bindings and counts that must exactly match the frozen config.
