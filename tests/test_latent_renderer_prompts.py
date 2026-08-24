import csv
import hashlib
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "eval-pipeline" / "prompts"


class LatentRendererPromptManifestTest(unittest.TestCase):
    def test_frozen_splits_match_manifest_and_are_disjoint(self):
        manifest = json.loads(
            (PROMPT_DIR / "latent_renderer_manifest.json").read_text()
        )
        excluded_rows = set(manifest["excluded_source_rows"])
        existing_texts = set()
        for path in PROMPT_DIR.glob("*.csv"):
            if path.name.startswith("latent_renderer_"):
                continue
            with path.open(newline="") as handle:
                existing_texts.update(row["TEXT"].strip() for row in csv.DictReader(handle))

        seen_rows = set()
        seen_prompts = set()
        challenges = {
            "Complex",
            "Fine-grained Detail",
            "Properties & Positioning",
            "Quantity",
            "Writing & Symbols",
            "Perspective",
        }
        for split in ("train", "validation", "test"):
            path = PROMPT_DIR / f"latent_renderer_{split}.csv"
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), manifest["counts"][split])
            self.assertEqual([int(row["index"]) for row in rows], list(range(len(rows))))
            self.assertTrue(all(row["split"] == split for row in rows))
            self.assertEqual(
                {row["source_challenge"] for row in rows}, challenges
            )
            for row in rows:
                source_row = int(row["source_row"])
                prompt = row["TEXT"].strip()
                self.assertNotIn(source_row, excluded_rows)
                self.assertNotIn(prompt, existing_texts)
                self.assertNotIn(source_row, seen_rows)
                self.assertNotIn(prompt, seen_prompts)
                seen_rows.add(source_row)
                seen_prompts.add(prompt)

        manifest_entries = {
            split: {
                (entry["source_row"], entry["prompt_sha256"])
                for entry in entries
            }
            for split, entries in manifest["splits"].items()
        }
        for split in ("train", "validation", "test"):
            with (PROMPT_DIR / f"latent_renderer_{split}.csv").open(
                newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            observed = {
                (
                    int(row["source_row"]),
                    hashlib.sha256(row["TEXT"].strip().encode()).hexdigest(),
                )
                for row in rows
            }
            self.assertEqual(observed, manifest_entries[split])

    def test_each_challenge_has_fixed_counts(self):
        for split, expected in (("train", 8), ("validation", 4), ("test", 4)):
            with (PROMPT_DIR / f"latent_renderer_{split}.csv").open(
                newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            counts = {}
            for row in rows:
                counts[row["source_challenge"]] = counts.get(
                    row["source_challenge"], 0
                ) + 1
            self.assertEqual(set(counts.values()), {expected})


if __name__ == "__main__":
    unittest.main()

