#!/usr/bin/env python3
"""
Benchmark multiple models against multiple prompts and write an interactive HTML report.

Example:
  python llm_benchmark.py \
    --url http://localhost:11434/api/generate \
    --models qwen3.8:27b llama3.1:8b \
    --prompts-file prompts.txt \
    --output benchmark.html

Alternatively, pass prompts directly:
  python llm_benchmark.py --url URL --models model-a model-b \
    --prompt "Why is the sky blue?" --prompt "Explain gravity."
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def ns_to_ms(value: Any) -> float | None:
    """Ollama reports durations in nanoseconds."""
    if not isinstance(value, (int, float)):
        return None
    return value / 1_000_000


def number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:,.2f}"


def tokens_per_second(token_count: Any, duration_ns: Any) -> float | None:
    if not isinstance(token_count, (int, float)):
        return None
    if not isinstance(duration_ns, (int, float)) or duration_ns <= 0:
        return None
    return token_count / (duration_ns / 1_000_000_000)


def read_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt or [])
    if args.prompts_file:
        path = Path(args.prompts_file)
        prompts.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    # Preserve order while removing exact duplicates.
    prompts = list(dict.fromkeys(p for p in prompts if p.strip()))
    if not prompts:
        raise SystemExit("No prompts supplied. Use --prompt or --prompts-file.")
    return prompts


def run_one(
    session: requests.Session,
    url: str,
    model: str,
    prompt: str,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    started = time.perf_counter()
    try:
        response = session.post(url, json=payload, timeout=timeout)
        wall_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            return {
                "model": model,
                "prompt": prompt,
                "ok": False,
                "wall_ms": wall_ms,
                "error": "Server returned a non-JSON response.",
                "raw_response": response.text,
            }

        if not isinstance(data, dict):
            return {
                "model": model,
                "prompt": prompt,
                "ok": False,
                "wall_ms": wall_ms,
                "error": "Expected a JSON object in the response.",
                "raw_response": json.dumps(data, ensure_ascii=False),
            }

        # Keep the useful fields; intentionally do not retain the potentially
        # very large "context" token array.
        return {
            "model": data.get("model", model),
            "prompt": prompt,
            "ok": True,
            "wall_ms": wall_ms,
            "created_at": data.get("created_at"),
            "response": data.get("response", ""),
            "thinking": data.get("thinking", ""),
            "total_duration": data.get("total_duration"),
            "load_duration": data.get("load_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_cached_count": data.get("prompt_eval_cached_count"),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "done": data.get("done"),
            "done_reason": data.get("done_reason"),
        }

    except requests.Timeout:
        return {
            "model": model,
            "prompt": prompt,
            "ok": False,
            "wall_ms": (time.perf_counter() - started) * 1000,
            "error": f"Request timed out after {timeout:g} seconds.",
        }
    except requests.RequestException as exc:
        return {
            "model": model,
            "prompt": prompt,
            "ok": False,
            "wall_ms": (time.perf_counter() - started) * 1000,
            "error": str(exc),
        }


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def details_block(title: str, content: Any) -> str:
    if not content:
        return ""
    return (
        '<details class="payload"><summary>'
        + esc(title)
        + '</summary><pre>'
        + esc(content)
        + "</pre></details>"
    )


def render_result(result: dict[str, Any], index: int) -> str:
    model = esc(result.get("model"))
    prompt = esc(result.get("prompt"))
    wall = number(result.get("wall_ms"))
    if not result.get("ok"):
        error = esc(result.get("error", "Unknown error"))
        raw = details_block("Raw response", result.get("raw_response"))
        return f"""
        <article class="result failed">
          <div class="result-head"><span class="model">{model}</span>
            <span class="status">FAILED</span><span class="muted">{wall} ms wall time</span></div>
          <div class="prompt"><b>Prompt:</b> {prompt}</div>
          <p class="error">{error}</p>{raw}
        </article>"""

    response_text = result.get("response") or ""
    thinking_text = result.get("thinking") or ""

    metrics = [
        ("Total", ns_to_ms(result.get("total_duration")), "ms"),
        ("Load", ns_to_ms(result.get("load_duration")), "ms"),
        ("Prompt eval", ns_to_ms(result.get("prompt_eval_duration")), "ms"),
        ("Generation", ns_to_ms(result.get("eval_duration")), "ms"),
        ("Prompt tokens", result.get("prompt_eval_count"), ""),
        ("Generated tokens", result.get("eval_count"), ""),
        ("Generation speed", tokens_per_second(
            result.get("eval_count"), result.get("eval_duration")
        ), "tok/s"),
        ("Wall time", result.get("wall_ms"), "ms"),
    ]
    metric_html = "".join(
        f'<div class="metric"><span>{esc(label)}</span><b>{number(value)} {unit}</b></div>'
        for label, value, unit in metrics
    )
    return f"""
    <article class="result">
      <div class="result-head"><span class="model">{model}</span>
        <span class="status success">OK</span>
        <span class="muted">{wall} ms wall time</span></div>
      <div class="prompt"><b>Prompt:</b> {prompt}</div>
      <div class="metrics">{metric_html}</div>
      {details_block("Response", response_text)}
      {details_block("Thinking", thinking_text)}
      <div class="small muted">Created: {esc(result.get("created_at") or "—")}
        · Done: {esc(result.get("done"))}
        · Reason: {esc(result.get("done_reason") or "—")}
        · Cached prompt tokens: {esc(result.get("prompt_eval_cached_count") or 0)}
      </div>
    </article>"""


def build_report(results: list[dict[str, Any]], url: str) -> str:
    successful = [r for r in results if r.get("ok")]
    failed = len(results) - len(successful)
    total_wall = sum(r.get("wall_ms", 0) for r in results)
    durations = [
        ns_to_ms(r.get("total_duration"))
        for r in successful
        if ns_to_ms(r.get("total_duration")) is not None
    ]
    avg_duration = statistics.mean(durations) if durations else None
    avg_speed_values = [
        tokens_per_second(r.get("eval_count"), r.get("eval_duration"))
        for r in successful
    ]
    avg_speed_values = [v for v in avg_speed_values if v is not None]
    avg_speed = statistics.mean(avg_speed_values) if avg_speed_values else None

    summary = [
        ("Requests", str(len(results))),
        ("Successful", str(len(successful))),
        ("Failed", str(failed)),
        ("Avg. model duration", f"{number(avg_duration)} ms"),
        ("Avg. generation speed", f"{number(avg_speed)} tok/s"),
        ("Combined wall time", f"{number(total_wall)} ms"),
    ]
    summary_html = "".join(
        f'<div class="summary-card"><span>{esc(label)}</span><b>{esc(value)}</b></div>'
        for label, value in summary
    )
    result_html = "\n".join(render_result(r, i) for i, r in enumerate(results))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Benchmark Report</title>
<style>
:root {{ color-scheme: light dark; --bg:#f4f6f9; --panel:#fff; --text:#18202b;
  --muted:#687386; --line:#dce2ea; --accent:#315fce; --bad:#b42318; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#11151c; --panel:#1b222d; --text:#edf1f7; --muted:#a5afbf;
    --line:#343e4d; --accent:#8eabff; --bad:#ff8b82; }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
  font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }}
main {{ max-width:1200px; margin:0 auto; padding:32px 20px 60px; }}
h1 {{ margin:0 0 5px; font-size:clamp(25px,4vw,36px); }}
.sub {{ color:var(--muted); overflow-wrap:anywhere; }}
.summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:12px; margin:24px 0 30px; }}
.summary-card,.result {{ background:var(--panel); border:1px solid var(--line);
  border-radius:12px; }}
.summary-card {{ padding:15px; }}
.summary-card span,.metric span {{ display:block; color:var(--muted); font-size:12px; }}
.summary-card b {{ display:block; font-size:21px; margin-top:4px; }}
.toolbar {{ display:flex; justify-content:space-between; align-items:center;
  gap:12px; margin:18px 0; flex-wrap:wrap; }}
button {{ border:1px solid var(--line); border-radius:8px; padding:8px 12px;
  background:var(--panel); color:var(--text); cursor:pointer; }}
.result {{ padding:18px; margin:14px 0; }}
.result.failed {{ border-color:var(--bad); }}
.result-head {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
.model {{ font-weight:700; font-size:17px; overflow-wrap:anywhere; }}
.status {{ font-size:11px; font-weight:700; padding:2px 7px; border-radius:20px;
  background:#ffebe9; color:var(--bad); }}
.status.success {{ background:#e6f4ea; color:#18743b; }}
.muted,.small {{ color:var(--muted); font-size:12px; }}
.prompt {{ margin:12px 0; white-space:pre-wrap; overflow-wrap:anywhere; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
  gap:8px; margin:14px 0; }}
.metric {{ background:var(--bg); border-radius:8px; padding:9px 11px; }}
.metric b {{ display:block; margin-top:3px; font-size:14px; }}
details.payload {{ border-top:1px solid var(--line); margin-top:12px; padding-top:10px; }}
details summary {{ cursor:pointer; font-weight:650; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word;
  font:13px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;
  background:var(--bg); padding:14px; border-radius:8px; max-height:650px;
  overflow:auto; }}
.error {{ color:var(--bad); overflow-wrap:anywhere; }}
</style>
</head>
<body><main>
<h1>LLM Benchmark Report</h1>
<div class="sub">Endpoint: {esc(url)}<br>Generated: {esc(generated)}
<br>Context/token arrays are excluded from this report.</div>
<section class="summary">{summary_html}</section>
<div class="toolbar"><h2>Individual results</h2>
  <div><button onclick="setDetails(true)">Expand all</button>
  <button onclick="setDetails(false)">Collapse all</button></div>
</div>
{result_html}
</main>
<script>
function setDetails(open) {{
  document.querySelectorAll('details').forEach(item => item.open = open);
}}
</script>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a model/prompt benchmark and generate an HTML report."
    )
    parser.add_argument("--url", required=True, help="POST endpoint URL")
    parser.add_argument("--models", nargs="+", required=True, help="One or more model names")
    parser.add_argument("--prompt", action="append", help="Prompt to test; may be repeated")
    parser.add_argument("--prompts-file", help="UTF-8 text file with one prompt per line")
    parser.add_argument("--output", default="benchmark_report.html", help="HTML output path")
    parser.add_argument("--timeout", type=float, default=300, help="Request timeout in seconds")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    prompts = read_prompts(args)
    results: list[dict[str, Any]] = []
    total = len(args.models) * len(prompts)

    with requests.Session() as session:
        session.headers.update({"Accept": "application/json"})
        for model in args.models:
            for prompt in prompts:
                index = len(results) + 1
                print(f"[{index}/{total}] model={model} | prompt={prompt[:70]!r}")
                result = run_one(session, args.url, model, prompt, args.timeout)
                results.append(result)
                if result["ok"]:
                    print(f"    OK — {number(result.get('wall_ms'))} ms wall time")
                else:
                    print(f"    FAILED — {result.get('error')}", file=sys.stderr)

    report = build_report(results, args.url)
    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {output_path.resolve()}")
    print(f"Completed: {len(results) - sum(not r['ok'] for r in results)}/{len(results)} successful")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
