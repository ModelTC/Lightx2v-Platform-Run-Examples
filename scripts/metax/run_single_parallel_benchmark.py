#!/usr/bin/env python3
"""Run independent MetaX single-card benchmarks concurrently, twice per case."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path("/data/Lightx2v-Platform-Run-Examples")
INFER = REPO / "scripts" / "metax" / "infer" / "single"
SUITES = REPO / "logs" / "metax" / "infer" / "suites"

CASES = (
    ("wan21_1_3b_t2v_480p_81f", "run_wan21_1_3b_t2v_480p_81f.sh"),
    ("longcat_image_t2i_1344x768", "run_longcat_image_t2i_1344x768.sh"),
    ("qwen_image_2512_t2i_1664x928", "run_qwen_image_2512_t2i_1664x928.sh"),
    ("flux2_dev_t2i_1344x768", "run_flux2_dev_t2i_1344x768.sh"),
    ("wan22_moe_a14b_t2v_480p_81f", "run_wan22_moe_a14b_t2v_480p_81f.sh"),
    ("wan22_moe_a14b_t2v_720p_81f", "run_wan22_moe_a14b_t2v_720p_81f.sh"),
    ("hunyuan_video_15_t2v_480p_121f", "run_hunyuan_video_15_t2v_480p_121f.sh"),
    ("hunyuan_video_15_t2v_720p_121f", "run_hunyuan_video_15_t2v_720p_121f.sh"),
    ("ltx2_3_22b_dev_s2v_768x512_241f", "run_ltx2_3_22b_dev_s2v_768x512_241f.sh"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    os.replace(temporary, path)


def load_record(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-id", required=True)
    args = parser.parse_args()
    suite_dir = SUITES / args.suite_id
    suite_dir.mkdir(parents=True, exist_ok=False)
    state_path = suite_dir / "suite.json"
    devices: queue.Queue[int] = queue.Queue()
    for device in range(8):
        devices.put(device)
    lock = threading.Lock()
    state: dict[str, Any] = {
        "suite_id": args.suite_id,
        "platform": "metax_cuda",
        "mode": "parallel_single_card_no_save",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "cases": {
            case_id: {"status": "pending", "runs": []}
            for case_id, _script in CASES
        },
    }
    atomic_json(state_path, state)

    def update() -> None:
        with lock:
            atomic_json(state_path, state)

    def run_case(case_index: int, case_id: str, script_name: str) -> bool:
        device = devices.get()
        case_state = state["cases"][case_id]
        case_state["device"] = device
        case_state["status"] = "running"
        update()
        all_ok = True
        try:
            for repetition in (1, 2):
                run_id = f"{args.suite_id}_{case_index:02d}_r{repetition}"
                record_path = (
                    REPO
                    / "logs"
                    / "metax"
                    / "infer"
                    / "single"
                    / case_id
                    / run_id
                    / "run.json"
                )
                controller_log = suite_dir / f"{case_index:02d}_{case_id}_r{repetition}.controller.log"
                run_entry = {
                    "repetition": repetition,
                    "run_id": run_id,
                    "started_at_utc": utc_now(),
                    "finished_at_utc": None,
                    "exit_code": None,
                    "run_record": str(record_path),
                    "controller_log": str(controller_log),
                }
                case_state["runs"].append(run_entry)
                update()
                env = os.environ.copy()
                env["RUN_ID"] = run_id
                env["CUDA_VISIBLE_DEVICES"] = str(device)
                with controller_log.open("wb") as output:
                    result = subprocess.run(
                        ["bash", str(INFER / script_name)],
                        cwd=REPO,
                        env=env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                record = load_record(record_path)
                metrics = record.get("metrics") if isinstance(record, dict) else {}
                metrics = metrics if isinstance(metrics, dict) else {}
                profile = metrics.get("profile_seconds")
                profile = profile if isinstance(profile, dict) else {}
                valid = (
                    result.returncode == 0
                    and record.get("status") == "succeeded"
                    and isinstance(metrics.get("dit_steady_state_seconds"), (int, float))
                    and metrics["dit_steady_state_seconds"] > 0
                    and isinstance(profile.get("pipeline"), (int, float))
                    and profile["pipeline"] > 0
                )
                run_entry.update(
                    {
                        "finished_at_utc": utc_now(),
                        "exit_code": result.returncode,
                        "valid": valid,
                        "steady_dit_seconds_per_step": metrics.get("dit_steady_state_seconds"),
                        "pipeline_cost_seconds": profile.get("pipeline"),
                    }
                )
                all_ok = all_ok and valid
                update()
                print(
                    f"[SingleParallel] {case_id} device={device} "
                    f"repetition={repetition} valid={valid}",
                    flush=True,
                )
        finally:
            case_state["status"] = "success" if all_ok else "failed"
            update()
            devices.put(device)
        return all_ok

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(run_case, index, case_id, script): case_id
            for index, (case_id, script) in enumerate(CASES, start=1)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                case_id = futures[future]
                state["cases"][case_id]["status"] = "failed"
                state["cases"][case_id]["error"] = repr(exc)
                update()

    failures = [
        case_id
        for case_id, case in state["cases"].items()
        if case["status"] != "success"
    ]
    state["status"] = "success" if not failures else "completed_with_failures"
    state["finished_at_utc"] = utc_now()
    measured = []
    for case_id, case in state["cases"].items():
        run = case["runs"][-1] if case["runs"] else {}
        measured.append(
            {
                "case_id": case_id,
                "status": case["status"],
                **{
                    key: run.get(key)
                    for key in (
                        "steady_dit_seconds_per_step",
                        "pipeline_cost_seconds",
                        "run_record",
                    )
                },
            }
        )
    state["second_run_results"] = measured
    atomic_json(state_path, state)
    atomic_json(
        suite_dir / "second_run_results.json",
        {
            "suite_id": args.suite_id,
            "repetition_used": 2,
            "cases": measured,
        },
    )
    print(
        f"[SingleParallel] status={state['status']} failures={len(failures)} "
        f"state={state_path}",
        flush=True,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
