"""
Sequential full Exp1 (phase1→2→3) + Exp2 for Qwen 3.5 0.8B base + instruct.
Does not touch any exp3 paths.

On step failure: one quick retry; if it fails again, skip and continue.
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PY if VENV_PY.exists() else sys.executable)
LOG_DIR = ROOT / "logs"
STATUS_PATH = LOG_DIR / "exp1_exp2_qwen08_status.txt"
MASTER_LOG = LOG_DIR / f"exp1_exp2_qwen08_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

DEVICE = "cuda"
MAX_RETRIES = 1

STEPS: list[tuple[str, str, str]] = [
    ("qwen_0.8b_base", "exp1_phase1", "src/models/qwen/0.8b_base/exp1/phase1_identify.py"),
    ("qwen_0.8b_base", "exp1_phase2", "src/models/qwen/0.8b_base/exp1/phase2_ablation.py"),
    ("qwen_0.8b_base", "exp1_phase3", "src/models/qwen/0.8b_base/exp1/phase3_distribution.py"),
    ("qwen_0.8b_base", "exp2", "src/models/qwen/0.8b_base/exp2/weight_projection.py"),
    ("qwen_0.8b_instruct", "exp1_phase1", "src/models/qwen/0.8b_instruct/exp1/phase1_identify.py"),
    ("qwen_0.8b_instruct", "exp1_phase2", "src/models/qwen/0.8b_instruct/exp1/phase2_ablation.py"),
    ("qwen_0.8b_instruct", "exp1_phase3", "src/models/qwen/0.8b_instruct/exp1/phase3_distribution.py"),
    ("qwen_0.8b_instruct", "exp2", "src/models/qwen/0.8b_instruct/exp2/weight_projection.py"),
]


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def write_status(payload: dict[str, object]) -> None:
    lines = [
        f"updated_at={_now()}",
        f"state={payload.get('state')}",
        f"model={payload.get('model')}",
        f"step={payload.get('step')}",
        f"attempt={payload.get('attempt')}",
        f"step_index={payload.get('step_index')}/{payload.get('step_total')}",
        f"step_log={payload.get('step_log')}",
        f"master_log={payload.get('master_log')}",
        f"pid={os.getpid()}",
        f"last_exit={payload.get('last_exit')}",
        f"message={payload.get('message')}",
        f"completed={payload.get('completed')}",
        f"skipped={payload.get('skipped')}",
        f"failed={payload.get('failed')}",
    ]
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_step(
    model: str,
    step: str,
    script: str,
    step_index: int,
    step_total: int,
    completed: list[str],
    skipped: list[str],
    failed: list[str],
) -> bool:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    step_log = LOG_DIR / f"step_{model}_{step}_{stamp}.log"
    cmd = [PYTHON, str(ROOT / script), "--device", DEVICE]

    for attempt in range(1, MAX_RETRIES + 2):
        header = (
            f"\n{'=' * 72}\n"
            f"[{_now()}] START model={model} step={step} attempt={attempt}/{MAX_RETRIES + 1}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"log: {step_log}\n"
            f"{'=' * 72}\n"
        )
        _append(MASTER_LOG, header)
        _append(step_log, header)
        write_status(
            {
                "state": "running",
                "model": model,
                "step": step,
                "attempt": attempt,
                "step_index": step_index,
                "step_total": step_total,
                "step_log": str(step_log),
                "master_log": str(MASTER_LOG),
                "last_exit": "",
                "message": "in progress",
                "completed": ",".join(completed) or "-",
                "skipped": ",".join(skipped) or "-",
                "failed": ",".join(failed) or "-",
            }
        )

        t0 = time.time()
        with step_log.open("a", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                write_status(
                    {
                        "state": "running",
                        "model": model,
                        "step": step,
                        "attempt": attempt,
                        "step_index": step_index,
                        "step_total": step_total,
                        "step_log": str(step_log),
                        "master_log": str(MASTER_LOG),
                        "last_exit": "",
                        "message": f"running pid={proc.pid} elapsed_s={int(time.time() - t0)}",
                        "completed": ",".join(completed) or "-",
                        "skipped": ",".join(skipped) or "-",
                        "failed": ",".join(failed) or "-",
                    }
                )
                time.sleep(30)

        elapsed = int(time.time() - t0)
        footer = f"[{_now()}] END model={model} step={step} exit={rc} elapsed_s={elapsed}\n"
        _append(MASTER_LOG, footer)
        _append(step_log, footer)

        if rc == 0:
            completed.append(f"{model}/{step}")
            write_status(
                {
                    "state": "running",
                    "model": model,
                    "step": step,
                    "attempt": attempt,
                    "step_index": step_index,
                    "step_total": step_total,
                    "step_log": str(step_log),
                    "master_log": str(MASTER_LOG),
                    "last_exit": rc,
                    "message": "ok",
                    "completed": ",".join(completed) or "-",
                    "skipped": ",".join(skipped) or "-",
                    "failed": ",".join(failed) or "-",
                }
            )
            return True

        try:
            tail = step_log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            _append(MASTER_LOG, "---- last 40 log lines ----\n" + "\n".join(tail) + "\n")
        except OSError as e:
            _append(MASTER_LOG, f"could not read step log: {e}\n")

        if attempt <= MAX_RETRIES:
            _append(MASTER_LOG, f"[{_now()}] retrying after failure (exit={rc})\n")
            time.sleep(5)
            continue

        failed.append(f"{model}/{step}")
        skipped.append(f"{model}/{step}")
        write_status(
            {
                "state": "skipping_after_failure",
                "model": model,
                "step": step,
                "attempt": attempt,
                "step_index": step_index,
                "step_total": step_total,
                "step_log": str(step_log),
                "master_log": str(MASTER_LOG),
                "last_exit": rc,
                "message": "failed after retries; skipping",
                "completed": ",".join(completed) or "-",
                "skipped": ",".join(skipped) or "-",
                "failed": ",".join(failed) or "-",
            }
        )
        return False

    return False


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    total = len(STEPS)

    _append(
        MASTER_LOG,
        f"[{_now()}] queue start: {total} steps, device={DEVICE}, python={PYTHON}\n",
    )
    write_status(
        {
            "state": "starting",
            "model": "-",
            "step": "-",
            "attempt": 0,
            "step_index": 0,
            "step_total": total,
            "step_log": "-",
            "master_log": str(MASTER_LOG),
            "last_exit": "",
            "message": "queue starting",
            "completed": "-",
            "skipped": "-",
            "failed": "-",
        }
    )

    for i, (model, step, script) in enumerate(STEPS, start=1):
        if step == "exp1_phase2" and f"{model}/exp1_phase1" in failed:
            skipped.append(f"{model}/{step}")
            _append(MASTER_LOG, f"[{_now()}] SKIP {model}/{step} (phase1 failed)\n")
            continue
        if step == "exp1_phase3" and (
            f"{model}/exp1_phase1" in failed or f"{model}/exp1_phase2" in failed
        ):
            skipped.append(f"{model}/{step}")
            _append(MASTER_LOG, f"[{_now()}] SKIP {model}/{step} (prior phase failed)\n")
            continue

        run_step(model, step, script, i, total, completed, skipped, failed)

    write_status(
        {
            "state": "finished",
            "model": "-",
            "step": "-",
            "attempt": 0,
            "step_index": total,
            "step_total": total,
            "step_log": "-",
            "master_log": str(MASTER_LOG),
            "last_exit": 0 if not failed else 1,
            "message": "queue complete",
            "completed": ",".join(completed) or "-",
            "skipped": ",".join(skipped) or "-",
            "failed": ",".join(failed) or "-",
        }
    )
    _append(
        MASTER_LOG,
        f"[{_now()}] queue finished. completed={len(completed)} skipped={len(skipped)} failed={len(failed)}\n",
    )
    print(f"DONE completed={len(completed)} skipped={len(skipped)} failed={len(failed)}")
    print(f"status={STATUS_PATH}")
    print(f"master_log={MASTER_LOG}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
