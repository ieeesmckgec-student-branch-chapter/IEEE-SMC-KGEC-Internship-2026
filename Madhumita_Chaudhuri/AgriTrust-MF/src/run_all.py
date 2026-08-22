"""End-to-end runner for the AgriTrust constrained-QAOA study.

Stages are declared with their resource needs so the orchestrator can use the
machine properly: the GPU sweep shards across every visible device, while the
CPU-bound studies run concurrently across the remaining cores. Every stage is
resumable, so an interrupted run can be restarted without repeating finished work.

    python run_all.py                 # everything, in dependency order
    python run_all.py --only 07 11    # just the scaling sweep and the noise study
    python run_all.py --skip 07       # everything except the GPU sweep
    python run_all.py --smoke         # fast reduced-size pass over every stage
    python run_all.py --list          # show the plan without running it
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
LOG_DIR = ROOT / "results" / "logs"
MANIFEST = ROOT / "results" / "run_manifest.json"


@dataclass
class Stage:
    key: str
    title: str
    command: list[str] | None = None
    # Stages that must finish first. Empty means it can start immediately.
    requires: tuple[str, ...] = ()
    # "gpu_sharded" fans out one worker per visible GPU, then runs `finalize`.
    kind: str = "cpu"
    finalize: list[str] | None = None
    smoke_command: list[str] | None = None
    smoke_finalize: list[str] | None = None
    expected_outputs: tuple[str, ...] = ()
    optional: bool = False
    notes: str = ""
    threads: int = 4


def script(name: str) -> list[str]:
    return [str(PYTHON), str(ROOT / "scripts" / name)]


STAGES: list[Stage] = [
    Stage(
        key="tests",
        title="Unit tests for every helper module",
        command=[str(PYTHON), "-m", "unittest", "discover", "-s", "tests", "-v"],
        notes="Gate: nothing else runs if the helpers are broken.",
        threads=8,
    ),
    Stage(
        key="00",
        title="Reproduce the published seed-7 notebook result",
        command=script("00_reproduce_seed7.py"),
        requires=("tests",),
        notes="Guards against refactoring drift in the ported pipeline.",
    ),
    Stage(
        key="03",
        title="Validate fast simulators against Qiskit Aer",
        command=script("03_validate_fast_sim.py"),
        requires=("tests",),
        notes="Every scaling number depends on this equivalence.",
    ),
    Stage(
        key="01",
        title="Audit the notebook's headline claims over 30 seeds",
        command=script("01_diagnose_claims.py"),
        requires=("00",),
        expected_outputs=("01_diagnostics_per_seed.csv", "01_risk_separability.csv"),
    ),
    Stage(
        key="02",
        title="Mixer x depth x objective ablation at 9 qubits",
        command=script("02_ablation_mixer_depth_objective.py"),
        requires=("00",),
        expected_outputs=("02_ablation.csv", "02_ablation_summary.csv"),
    ),
    Stage(
        key="04",
        title="Detection versus adversary stealth",
        command=script("04_detection_under_stealth.py"),
        requires=("tests",),
        expected_outputs=("04_detection_under_stealth.csv",),
    ),
    Stage(
        key="05",
        title="Hardness ladder rung 1: layered DAG is polynomial",
        command=script("05_classical_hardness.py"),
        requires=("tests",),
        expected_outputs=("05_classical_hardness.csv",),
    ),
    Stage(
        key="06",
        title="Hardness ladder rung 2: interference breaks the DP",
        command=script("06_interference_hardness.py"),
        requires=("tests",),
        expected_outputs=("06_interference_hardness.csv",),
    ),
    Stage(
        key="08",
        title="Heavy-hex hardware-basis depth and CX cost",
        command=script("08_hardware_basis_cost.py"),
        requires=("tests",),
        expected_outputs=("08_hardware_basis_raw.csv", "08_hardware_basis_summary.csv"),
    ),
    Stage(
        key="07",
        title="Scaling sweep 9->25 qubits, XY versus penalty",
        kind="gpu_sharded",
        command=script("07_scaling_sweep.py") + ["worker"],
        finalize=script("07_scaling_sweep.py") + ["summarize"],
        smoke_command=script("07_scaling_sweep.py")
        + ["worker", "--seeds", "2", "--sizes", "3x3,4x4", "--restarts", "1", "--maxiter", "20"],
        smoke_finalize=script("07_scaling_sweep.py")
        + ["summarize", "--seeds", "2", "--sizes", "3x3,4x4"],
        requires=("03",),
        expected_outputs=("07_scaling_raw.csv", "07_scaling_summary.csv"),
        notes="The core contribution figure. Shards across every visible GPU.",
    ),
    Stage(
        key="09",
        title="Multi-round network lifetime replacing one-shot PDR",
        command=script("09_network_lifetime.py"),
        smoke_command=script("09_network_lifetime.py") + ["--seeds", "2", "--rounds", "5"],
        requires=("tests",),
        optional=True,
    ),
    Stage(
        key="10",
        title="Stage B: routing-metric poisoning and robust QUBO",
        command=script("10_poisoning_robustness.py"),
        smoke_command=script("10_poisoning_robustness.py") + ["--seeds", "2"],
        requires=("tests",),
        optional=True,
    ),
    Stage(
        key="11",
        title="Noise-induced feasibility decay and post-selection cost",
        command=script("11_noise_postselection.py"),
        smoke_command=script("11_noise_postselection.py") + ["--seeds", "2"],
        requires=("03",),
        optional=True,
    ),
    Stage(
        key="12",
        title="Hardness ladder rung 3: multi-flow congestion (NP-hard)",
        command=script("12_multiflow_hardness.py"),
        smoke_command=script("12_multiflow_hardness.py") + ["--seeds", "2"],
        requires=("tests",),
        optional=True,
    ),
]

STAGE_BY_KEY = {stage.key: stage for stage in STAGES}


def visible_gpus() -> list[str]:
    override = os.environ.get("CUDA_VISIBLE_DEVICES")
    if override:
        return [item for item in override.split(",") if item != ""]
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def thread_env(threads: int) -> dict[str, str]:
    """Pin BLAS thread counts so concurrent stages do not oversubscribe cores."""
    env = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = str(threads)
    return env


@dataclass
class Running:
    stage: Stage
    processes: list[subprocess.Popen] = field(default_factory=list)
    handles: list = field(default_factory=list)
    started: float = 0.0
    phase: str = "main"


def launch(stage: Stage, smoke: bool, gpus: list[str]) -> Running:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    running = Running(stage=stage, started=time.perf_counter())

    command = stage.smoke_command if (smoke and stage.smoke_command) else stage.command
    if command is None:
        raise ValueError(f"Stage {stage.key} has no command")

    if stage.kind == "gpu_sharded" and gpus:
        shards = len(gpus)
        for index, gpu in enumerate(gpus):
            env = thread_env(stage.threads)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log = (LOG_DIR / f"{stage.key}_shard{index}.log").open("w")
            running.handles.append(log)
            running.processes.append(
                subprocess.Popen(
                    command + ["--shard", str(index), "--shards", str(shards)],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            )
    else:
        log = (LOG_DIR / f"{stage.key}.log").open("w")
        running.handles.append(log)
        running.processes.append(
            subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=thread_env(stage.threads),
            )
        )
    return running


def finalize(stage: Stage, smoke: bool) -> int:
    command = stage.smoke_finalize if (smoke and stage.smoke_finalize) else stage.finalize
    if command is None:
        return 0
    log = LOG_DIR / f"{stage.key}_summarize.log"
    with log.open("w") as handle:
        return subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=thread_env(8),
        ).returncode


def tail(path: Path, lines: int = 12) -> str:
    if not path.exists():
        return "(no log)"
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Stage keys to run")
    parser.add_argument("--skip", nargs="*", default=(), help="Stage keys to skip")
    parser.add_argument("--smoke", action="store_true", help="Fast reduced-size pass")
    parser.add_argument("--list", action="store_true", help="Print the plan and exit")
    parser.add_argument(
        "--max-parallel", type=int, default=4, help="Concurrent CPU stages"
    )
    args = parser.parse_args()

    selected = [
        stage
        for stage in STAGES
        if (args.only is None or stage.key in args.only) and stage.key not in args.skip
    ]

    if args.list:
        gpus = visible_gpus()
        print(f"{len(selected)} stages, {len(gpus)} GPU(s) visible\n")
        for stage in selected:
            marker = "optional" if stage.optional else "required"
            requires = ",".join(stage.requires) or "-"
            print(f"  {stage.key:<6} {stage.title}")
            print(f"         kind={stage.kind} requires={requires} ({marker})")
        return 0

    gpus = visible_gpus()
    print(f"orchestrator: {len(selected)} stages, GPUs visible: {gpus or 'none'}")
    if args.smoke:
        print("mode: SMOKE (reduced sizes, not publication data)")
    print()

    pending = list(selected)
    running: list[Running] = []
    done: dict[str, dict] = {}
    started_at = time.perf_counter()

    def ready(stage: Stage) -> bool:
        for requirement in stage.requires:
            if requirement not in STAGE_BY_KEY:
                continue
            if requirement in {item.key for item in selected}:
                record = done.get(requirement)
                if record is None or record["status"] != "ok":
                    return False
        return True

    while pending or running:
        # Launch whatever is ready, respecting the parallel budget.
        while pending:
            candidates = [stage for stage in pending if ready(stage)]
            if not candidates:
                break
            gpu_busy = any(item.stage.kind == "gpu_sharded" for item in running)
            stage = None
            for candidate in candidates:
                if candidate.kind == "gpu_sharded":
                    if gpu_busy:
                        continue
                    stage = candidate
                    break
            if stage is None:
                cpu_running = sum(1 for item in running if item.stage.kind != "gpu_sharded")
                if cpu_running >= args.max_parallel:
                    break
                cpu_candidates = [c for c in candidates if c.kind != "gpu_sharded"]
                if not cpu_candidates:
                    break
                stage = cpu_candidates[0]

            missing_script = False
            if stage.command:
                if "-m" in stage.command:
                    missing_script = False
                elif stage.kind == "gpu_sharded":
                    missing_script = not Path(stage.command[-2]).exists()
                else:
                    missing_script = not Path(stage.command[1]).exists()
            if missing_script:
                message = f"script not found: {stage.command}"
                status = "skipped" if stage.optional else "failed"
                print(f"[{status:>7}] {stage.key} {stage.title}\n          {message}")
                done[stage.key] = {"status": status, "reason": message, "seconds": 0.0}
                pending.remove(stage)
                continue

            pending.remove(stage)
            shard_note = f" across {len(gpus)} GPU(s)" if stage.kind == "gpu_sharded" else ""
            print(f"[ start ] {stage.key} {stage.title}{shard_note}")
            running.append(launch(stage, args.smoke, gpus))

        if not running:
            if pending:
                blocked = ", ".join(stage.key for stage in pending)
                print(f"\nstopping: unmet dependencies for {blocked}")
            break

        time.sleep(2)

        for item in list(running):
            if any(process.poll() is None for process in item.processes):
                continue

            codes = [process.returncode for process in item.processes]
            for handle in item.handles:
                handle.close()
            item.handles.clear()

            if item.phase == "main" and all(code == 0 for code in codes) and item.stage.finalize:
                item.phase = "finalize"
                code = finalize(item.stage, args.smoke)
                codes.append(code)

            elapsed = time.perf_counter() - item.started
            ok = all(code == 0 for code in codes)
            missing = [
                name
                for name in item.stage.expected_outputs
                if not (ROOT / "results" / name).exists()
            ]
            if ok and missing:
                ok = False

            status = "ok" if ok else ("skipped" if item.stage.optional else "failed")
            done[item.stage.key] = {
                "status": status,
                "seconds": elapsed,
                "returncodes": codes,
                "missing_outputs": missing,
            }
            running.remove(item)

            label = "  done  " if ok else ("skipped" if item.stage.optional else " FAILED")
            print(f"[{label}] {item.stage.key} {item.stage.title}  ({elapsed / 60:.1f} min)")
            if not ok:
                log = LOG_DIR / (
                    f"{item.stage.key}_shard0.log"
                    if item.stage.kind == "gpu_sharded"
                    else f"{item.stage.key}.log"
                )
                print(f"          returncodes={codes} missing={missing}")
                print("          last log lines:")
                for line in tail(log).splitlines():
                    print(f"            {line}")

    total = time.perf_counter() - started_at
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {"smoke": args.smoke, "total_seconds": total, "stages": done},
            indent=2,
            sort_keys=True,
        )
    )

    ok = [key for key, record in done.items() if record["status"] == "ok"]
    skipped = [key for key, record in done.items() if record["status"] == "skipped"]
    failed = [key for key, record in done.items() if record["status"] == "failed"]

    print(f"\n{'=' * 72}")
    print(f"TOTAL {total / 60:.1f} min   ok={len(ok)} skipped={len(skipped)} failed={len(failed)}")
    print("=" * 72)
    if skipped:
        print(f"skipped (optional): {', '.join(sorted(skipped))}")
    if failed:
        print(f"FAILED: {', '.join(sorted(failed))}")
    print(f"manifest: {MANIFEST}")
    print(f"logs:     {LOG_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
