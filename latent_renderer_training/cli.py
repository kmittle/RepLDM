"""Small contract probe for the latent-renderer training package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .contracts import ActionSpaceContract
from .ledger import QueryLedger
from .launcher import launch_training


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # ``train`` is the only formal training entry point.  Keep the historical
    # dependency-free probe available for audits and smoke checks.
    if argv and argv[0] == "train":
        return training_main(argv[1:])
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


def training_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m latent_renderer_training.cli train",
        description="Validate and optionally dispatch one authorized renderer run",
    )
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = launch_training(
        receipt_path=args.receipt,
        config_path=args.config,
        repository_root=args.repository_root,
        validation_only=args.validate_only,
    )
    print(
        json.dumps(
            {
                "schema": "repldm.training_launch.v1",
                "run_id": result.run_id,
                "method": result.method,
                "run_contract_sha256": result.run_contract_sha256,
                "config_sha256": result.config_sha256,
                "authorization_receipt": result.authorization_receipt,
                "executed": result.executed,
                "validation_only": result.validation_only,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
