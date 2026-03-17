#!/usr/bin/env python3
"""Minimal polling client for the external agent run API."""
from __future__ import annotations

import os
import sys
import time

import requests


BACKEND_URL = os.getenv("LOCAL_COCOA_BACKEND_URL", "http://127.0.0.1:8890")
API_KEY = os.getenv("LOCAL_COCOA_API_KEY", "sk-ty3qjw-D-33eMyIt_eBLZ1YsK5EDHSDmhsv2N8mfWJk")


def _headers() -> dict[str, str]:
    if not API_KEY:
        raise RuntimeError("Set LOCAL_COCOA_API_KEY before running this script.")
    return {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
        "X-Request-Source": "external",
    }


def main() -> int:
    query = "check SYNVO AI Compnay information and send email summary to my email."

    create_response = requests.post(
        f"{BACKEND_URL}/agent/external/runs",
        headers=_headers(),
        json={
            "query": query,
            "conversation_history": [],
            "max_iterations": 10,
        },
        timeout=30,
    )
    create_response.raise_for_status()
    created = create_response.json()
    run_id = created["run_id"]

    print(f"Started run: {run_id}")

    last_seq = 0
    seen_done = False

    while True:
        events_response = requests.get(
            f"{BACKEND_URL}/agent/external/runs/{run_id}/events",
            headers=_headers(),
            params={"after_seq": last_seq, "limit": 100},
            timeout=30,
        )
        events_response.raise_for_status()
        events_payload = events_response.json()

        for record in events_payload.get("events", []):
            last_seq = max(last_seq, int(record["seq"]))
            event = record["event"]
            event_type = event["type"]
            data = event.get("data")

            if event_type == "thinking_step" and isinstance(data, dict):
                print(f"[thinking] {data.get('title')}")
            elif event_type == "tool_call" and isinstance(data, dict):
                print(f"[tool_call] {data.get('tool')} args={data.get('args')}")
            elif event_type == "tool_result" and isinstance(data, dict):
                print(f"[tool_result] {data.get('tool')} success={data.get('success')}")
                preview = data.get("output_preview")
                if preview:
                    print(f"  preview: {preview}")
            elif event_type == "error":
                print(f"[error] {data}")
            elif event_type == "done":
                seen_done = True

        state_response = requests.get(
            f"{BACKEND_URL}/agent/external/runs/{run_id}",
            headers=_headers(),
            timeout=30,
        )
        state_response.raise_for_status()
        state = state_response.json()

        print(f"[status] {state['status']} phase={state['current_phase']} message={state.get('latest_message')}")

        if state["status"] in {"completed", "failed"} or seen_done:
            print("\nFinal answer:\n")
            print(state.get("final_answer") or "<empty>")
            if state.get("error"):
                print(f"\nError: {state['error']}", file=sys.stderr)
                return 1
            return 0

        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
