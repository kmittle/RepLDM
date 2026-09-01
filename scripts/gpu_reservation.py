#!/usr/bin/env python3
"""Reserve selected GPUs while no formal experiment is running.

The workers allocate nearly all currently free memory and continuously execute
an FP16 matrix multiply. This is an operational resource lock, not an
experiment: stop it before generation, training, or scoring, and restart it
while a finished run is being debugged.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

MARKER = "repldm-gpu-reservation-v1"
DEFAULT_STATE_DIR = Path("/tmp/repldm_gpu_reservation")
MIB = 1024 * 1024


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError):
        return ""
    return raw.replace(b"\0", b" " ).decode("utf-8", errors="replace").strip()


def _is_worker(pid: int) -> bool:
    if not _pid_alive(pid):
        return False
    command = _cmdline(pid)
    return "gpu_reservation.py worker" in command and MARKER in command


def _parse_devices(values: Iterable[str]) -> list[int]:
    devices: list[int] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            device = int(item)
            if device < 0:
                raise ValueError(f"GPU index must be non-negative: {device}")
            if device not in devices:
                devices.append(device)
    if not devices:
        raise ValueError("at least one GPU is required")
    return devices


def _allocate_matmul(torch: Any, device: Any, requested_n: int) -> tuple[Any, Any, Any, int]:
    n = requested_n
    if n < 1024 or n % 256:
        raise ValueError("--matmul-size must be a multiple of 256 and at least 1024")
    while n >= 1024:
        try:
            a = torch.empty((n, n), dtype=torch.float16, device=device)
            b = torch.empty((n, n), dtype=torch.float16, device=device)
            c = torch.empty((n, n), dtype=torch.float16, device=device)
            a.uniform_(-1.0, 1.0)
            b.uniform_(-1.0, 1.0)
            return a, b, c, n
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or n == 1024:
                raise
            torch.cuda.empty_cache()
            n -= 256
    raise RuntimeError("unable to allocate matrix operands")


def _allocate_reserve(torch: Any, free_bytes: int, margin_mib: int) -> tuple[Any, int]:
    target = max(0, int(free_bytes) - margin_mib * MIB)
    decrement = max(64 * MIB, margin_mib * MIB // 2)
    while target >= 0:
        try:
            reserve = torch.empty((target,), dtype=torch.uint8, device="cuda:0")
            reserve.fill_(0)
            return reserve, target
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            target -= decrement
    raise RuntimeError("unable to allocate GPU reservation block")


def worker_main(physical_device: int, state_dir: Path, matmul_size: int, margin_mib: int) -> int:
    import torch

    if os.environ.get("REPLDM_GPU_RESERVATION_MARKER") != MARKER:
        raise RuntimeError("worker must be launched by the reservation controller")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_device):
        raise RuntimeError("worker CUDA_VISIBLE_DEVICES does not match its physical GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    a, b, c, actual_n = _allocate_matmul(torch, device, matmul_size)
    torch.cuda.synchronize()
    free_after_operands, total_bytes = torch.cuda.mem_get_info(device)
    reserve, reserved_bytes = _allocate_reserve(torch, free_after_operands, margin_mib)
    torch.cuda.synchronize()

    state_path = state_dir / f"worker-{physical_device}.json"
    _json_dump(
        state_path,
        {
            "marker": MARKER,
            "pid": os.getpid(),
            "physical_device": physical_device,
            "visible_device": 0,
            "gpu_name": torch.cuda.get_device_name(0),
            "total_mib": int(total_bytes // MIB),
            "reserved_mib": int(reserved_bytes // MIB),
            "matmul_size": actual_n,
            "started_at": time.time(),
        },
    )
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        iteration = 0
        while not stop:
            torch.mm(a, b, out=c)
            iteration += 1
            if iteration % 8 == 0:
                torch.cuda.synchronize()
    finally:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            state_path.unlink()
        except FileNotFoundError:
            pass
        del reserve, a, b, c
        torch.cuda.empty_cache()
    return 0


def _worker_pid_path(state_dir: Path, device: int) -> Path:
    return state_dir / f"worker-{device}.pid"


def start(args: argparse.Namespace) -> int:
    state_dir = args.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    devices = _parse_devices(args.devices)
    records: list[dict[str, Any]] = []
    for device in devices:
        pid_path = _worker_pid_path(state_dir, device)
        if pid_path.exists():
            try:
                old_pid = int(pid_path.read_text(encoding="utf-8").strip())
            except ValueError:
                old_pid = -1
            if old_pid > 0 and _is_worker(old_pid):
                print(f"GPU {device}: already reserved by PID {old_pid}")
                records.append({"device": device, "pid": old_pid, "status": "existing"})
                continue
            pid_path.unlink(missing_ok=True)
        log_path = state_dir / f"worker-{device}.log"
        log_handle = log_path.open("ab")
        environment = os.environ.copy()
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.pop("CUDA_DEVICE_ORDER", None)
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
        environment["REPLDM_GPU_RESERVATION_MARKER"] = MARKER
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--device",
            str(device),
            "--state-dir",
            str(state_dir),
            "--matmul-size",
            str(args.matmul_size),
            "--reserve-margin-mib",
            str(args.reserve_margin_mib),
            "--marker",
            MARKER,
        ]
        process = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
        log_handle.close()
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
        records.append({"device": device, "pid": process.pid, "status": "started"})
        print(f"GPU {device}: started reservation worker PID {process.pid}")
    _json_dump(
        state_dir / "controller.json",
        {
            "marker": MARKER,
            "devices": devices,
            "workers": records,
            "started_at": time.time(),
            "matmul_size": args.matmul_size,
            "reserve_margin_mib": args.reserve_margin_mib,
        },
    )
    time.sleep(args.ready_wait_seconds)
    return status(argparse.Namespace(state_dir=state_dir, devices=[str(d) for d in devices]))


def stop(args: argparse.Namespace) -> int:
    state_dir = args.state_dir
    devices = _parse_devices(args.devices)
    stopped = 0
    for device in devices:
        pid_path = _worker_pid_path(state_dir, device)
        if not pid_path.exists():
            print(f"GPU {device}: no reservation PID found")
            continue
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = -1
        if pid <= 0 or not _is_worker(pid):
            print(f"GPU {device}: stale or unrecognized PID {pid}; leaving it untouched")
            pid_path.unlink(missing_ok=True)
            continue
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + args.timeout_seconds
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        if _pid_alive(pid):
            print(f"GPU {device}: PID {pid} did not exit; sending SIGKILL")
            os.kill(pid, signal.SIGKILL)
        else:
            print(f"GPU {device}: stopped reservation PID {pid}")
        pid_path.unlink(missing_ok=True)
        stopped += 1
    return 0 if stopped or not devices else 1


def status(args: argparse.Namespace) -> int:
    state_dir = args.state_dir
    devices = _parse_devices(args.devices)
    for device in devices:
        pid_path = _worker_pid_path(state_dir, device)
        if not pid_path.exists():
            print(f"GPU {device}: free (no reservation PID)")
            continue
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = -1
        state_path = state_dir / f"worker-{device}.json"
        details = {}
        if state_path.exists():
            try:
                details = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                details = {"state": "invalid-json"}
        if pid > 0 and _is_worker(pid):
            print(
                f"GPU {device}: reserved PID {pid}, "
                f"memory={details.get('reserved_mib', '?')} MiB, "
                f"matmul={details.get('matmul_size', '?')}"
            )
        else:
            print(f"GPU {device}: stale reservation PID {pid}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--devices", nargs="+", default=["4", "5", "6", "7"],
            help="physical GPU indices (space or comma separated; default: 4 5 6 7)",
        )
        subparser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)

    start_parser = subparsers.add_parser("start", help="start marked GPU reservation workers")
    common(start_parser)
    start_parser.add_argument("--matmul-size", type=int, default=8192)
    start_parser.add_argument("--reserve-margin-mib", type=int, default=128)
    start_parser.add_argument("--ready-wait-seconds", type=float, default=4.0)
    start_parser.set_defaults(handler=start)

    stop_parser = subparsers.add_parser("stop", help="stop only marked reservation workers")
    common(stop_parser)
    stop_parser.add_argument("--timeout-seconds", type=float, default=10.0)
    stop_parser.set_defaults(handler=stop)

    status_parser = subparsers.add_parser("status", help="show reservation status")
    common(status_parser)
    status_parser.set_defaults(handler=status)

    worker_parser = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--device", type=int, required=True)
    worker_parser.add_argument("--state-dir", type=Path, required=True)
    worker_parser.add_argument("--matmul-size", type=int, default=8192)
    worker_parser.add_argument("--reserve-margin-mib", type=int, default=384)
    worker_parser.add_argument("--marker", required=True)
    worker_parser.set_defaults(
        handler=lambda worker_args: worker_main(
            worker_args.device, worker_args.state_dir, worker_args.matmul_size,
            worker_args.reserve_margin_mib
        )
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
