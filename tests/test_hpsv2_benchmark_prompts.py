from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "eval-pipeline/prompts/hpsv2_official_3200.csv"
MANIFEST_PATH = (
    ROOT / "eval-pipeline/prompts/hpsv2_official_3200_manifest.json"
)


class HPSv2BenchmarkPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with CSV_PATH.open(encoding="utf-8", newline="") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="ascii"))

    def test_complete_official_design(self) -> None:
        self.assertEqual(len(self.rows), 3200)
        self.assertEqual([int(row["index"]) for row in self.rows], list(range(3200)))
        self.assertEqual(
            [self.rows[offset]["benchmark_style"] for offset in (0, 800, 1600, 2400)],
            ["anime", "concept-art", "paintings", "photo"],
        )
        for style in self.manifest["style_order"]:
            selected = [row for row in self.rows if row["benchmark_style"] == style]
            self.assertEqual(len(selected), 800)
            self.assertEqual(
                [int(row["style_index"]) for row in selected], list(range(800))
            )
            self.assertEqual(
                [row["source_row_id"] for row in selected],
                [f"{style}:{index:04d}" for index in range(800)],
            )

    def test_manifest_binds_csv_and_preserves_duplicates(self) -> None:
        raw = CSV_PATH.read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), self.manifest["csv"]["sha256"]
        )
        unique_count = len({row["TEXT"] for row in self.rows})
        self.assertEqual(unique_count, 3172)
        self.assertEqual(self.manifest["exact_unique_prompt_count"], unique_count)
        self.assertEqual(self.manifest["official_duplicate_row_count"], 28)
        counts = Counter(row["TEXT"] for row in self.rows)
        self.assertEqual(sum(count > 1 for count in counts.values()), 23)
        self.assertEqual(max(counts.values()), 4)
        self.assertFalse(any(not row["TEXT"].strip() for row in self.rows))

    def test_source_revision_and_style_hashes_are_frozen(self) -> None:
        source = self.manifest["source"]
        self.assertEqual(
            source["revision"], "b9517430f34b1080d4f741118d2a440155a165cd"
        )
        self.assertEqual(
            {style: source["files"][style]["sha256"] for style in source["files"]},
            {
                "anime": "6b38a479cc6f899866e9b7a8bf91f096b4afc6883b95275a61891327182b14ae",
                "concept-art": "530d057deafcb82c581e9d1be2ffcb98828fe99f0b54257414d2ab8a5cf2b83e",
                "paintings": "6570d39467e08b24b64e5b8a4177b0f61c525631121dc5a91fe2ab956eab4d5f",
                "photo": "92a3c9a0f743f66eb87bd112cdda830d6bdb4c073c86021d53b74001fcb18052",
            },
        )


if __name__ == "__main__":
    unittest.main()
