"""Small contract probe for the latent-renderer training package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import ActionSpaceContract
from .ledger import QueryLedger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--ledger", default=None)
    args = parser.parse_args(argv)
    if args.slots <= 0:
        raise ValueError("--slots must be positive")
    contract = ActionSpaceContract(args.slots, tuple(True for _ in range(args.slots)))
    result = {"schema": "repldm.training_probe.v1", "contract": contract.to_dict()}
    if args.ledger:
        # A probe is read-only.  In particular, it must not create a ledger
        # directory or reserve a query merely because a path was supplied.
        ledger_path = Path(args.ledger)
        if ledger_path.exists():
            ledger = QueryLedger(
                ledger_path,
                {"reward_forward": 1},
                run_contract="probe",
                strict_provenance=True,
            )
            result["ledger"] = ledger.summary()
        else:
            result["ledger"] = {
                "path": str(ledger_path),
                "exists": False,
                "read_only": True,
            }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
