"""
Gradio app for vLLM Optima.

Two ways to use it:
  - Run live:      needs an NVIDIA GPU + vLLM (e.g. Colab T4). Calls agent.run_agent().
  - Watch a run:   replays real recorded runs from recordings/*.json. Works without a GPU.

Colab:   import app; app.demo.queue().launch(share=True)
Local:   python app.py
"""
import os, json, time, html, subprocess
from pathlib import Path

import gradio as gr

from agent import run_agent, clean_error, optimize

APP_NAME = "vLLM Optima"
TAGLINE = "Give it any model: it fixes what breaks, makes it faster, and proves every change on your GPU."
HERE = Path(__file__).parent
RECORDINGS = HERE / "recordings"
RUNS = HERE / "runs"


def _nvidia_smi(*args):
    for exe in ("nvidia-smi", "/opt/bin/nvidia-smi", "/usr/bin/nvidia-smi"):
        try:
            p = subprocess.run([exe, *args], capture_output=True, text=True, timeout=10)
            if p.returncode == 0:
                return p.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def gpu_available():
    """True if an NVIDIA GPU answers (checked each time, not only when the app starts)."""
    return _nvidia_smi("-L") is not None


def gpu_label():
    out = _nvidia_smi("--query-gpu=name,memory.total", "--format=csv,noheader")
    if not out:
        return None
    lines = out.splitlines()
    name, mem = [x.strip() for x in lines[0].split(",")]
    return f"{name}{' x' + str(len(lines)) if len(lines) > 1 else ''} · {mem}"


GPU_LABEL = gpu_label()

# ------------------------------------------------------------------ look and feel
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {
  --ink: #16212A; --muted: #5E6E7A; --line: #DCE2E6; --paper: #EEF1F3; --panel: #FFFFFF;
  --run: #C98A12; --ok: #1D7A5A; --fail: #B43F2E; --brain: #3A55A4; --info: #7A8A96; --warn: #9A6B00; --web: #0F7C8C;
}
.gradio-container { font-family: 'IBM Plex Sans', system-ui, sans-serif !important; background: var(--paper) !important;
  max-width: 1280px !important; }
footer { display: none !important; }

/* app bar */
.va-bar { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
  background: var(--ink); color: #fff; padding: 0.9rem 1.2rem; border-radius: 10px; }
.va-bar .name { font-size: 1.25rem; font-weight: 600; letter-spacing: -0.01em; }
.va-bar .tag { color: #C9D3DA; font-size: 0.92rem; margin-top: 0.1rem; }
.va-pills { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.va-pill { font-size: 0.82rem; padding: 0.25rem 0.6rem; border-radius: 999px; background: rgba(255,255,255,0.1);
  color: #fff !important; white-space: nowrap; }
.va-bar .name { color: #fff !important; }
.va-bar .tag { color: #C9D3DA !important; }
.va-pill .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 0.4rem;
  background: var(--c, #9AA7B1); vertical-align: middle; }

/* panels */
.va-panel { background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 10px !important;
  padding: 1rem !important; }
.va-panel-title { font-weight: 600; color: var(--ink); margin: 0 0 0.2rem; }
.va-hint { color: var(--muted); font-size: 0.9rem; margin: 0 0 0.6rem; }
.va-examples button { justify-content: flex-start !important; text-align: left !important; }

/* run header */
.va-runhead { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; flex-wrap: wrap;
  padding-bottom: 0.7rem; margin-bottom: 0.6rem; border-bottom: 1px solid var(--line); }
.va-runhead .what { font-weight: 600; color: var(--ink); overflow-wrap: anywhere; }
.va-runhead .meta { color: var(--muted); font-size: 0.88rem; }
.va-status { font-size: 0.82rem; font-weight: 500; padding: 0.2rem 0.6rem; border-radius: 999px; white-space: nowrap;
  color: var(--c); background: color-mix(in srgb, var(--c) 12%, white); }

/* trace */
ol.va-trace, ol.va-trace > li { list-style: none !important; margin-left: 0 !important; }
ol.va-trace { padding: 0.2rem 0 0 0 !important; margin: 0 !important; position: relative; }
ol.va-trace > li::marker { content: none !important; }
ol.va-trace::before { content: ""; position: absolute; left: 7px; top: 0.6rem; bottom: 1.2rem; width: 2px; background: var(--line); }
.va-step { position: relative; padding: 0 0 1rem 1.9rem !important; }
.va-step::before { content: ""; position: absolute; left: 0; top: 0.3rem; width: 16px; height: 16px;
  border-radius: 50%; background: var(--panel); border: 3px solid var(--c); box-sizing: border-box; }
.va-step.active::before { background: var(--c); animation: va-pulse 1.4s ease-in-out infinite; }
@keyframes va-pulse { 50% { box-shadow: 0 0 0 6px color-mix(in srgb, var(--c) 22%, transparent); } }
@media (prefers-reduced-motion: reduce) { .va-step.active::before { animation: none; } }
.va-step.info, .va-status.info { --c: var(--info); } .va-step.run, .va-status.run { --c: var(--run); }
.va-step.ok, .va-status.ok { --c: var(--ok); } .va-step.fail, .va-status.fail { --c: var(--fail); }
.va-step.brain, .va-status.brain { --c: var(--brain); } .va-step.warn { --c: var(--warn); }
.va-step.web, .va-status.web { --c: var(--web); }
.va-title { font-weight: 600; color: var(--ink); }
.va-title .who { font-weight: 500; color: var(--brain); margin-left: 0.3rem; }
.va-time { color: var(--muted); font-size: 0.82em; margin-left: 0.5rem; font-variant-numeric: tabular-nums; }
.va-text { color: var(--ink); margin-top: 0.15rem; line-height: 1.45; overflow-wrap: anywhere; }
.va-step.brain .va-text { border-left: 3px solid var(--brain); padding-left: 0.6rem; }
.va-chips { margin-top: 0.4rem; display: flex; flex-wrap: wrap; gap: 0.35rem; }
.va-chip { font-family: 'IBM Plex Mono', monospace; font-size: 0.8em; padding: 0.1rem 0.45rem; border-radius: 4px;
  background: color-mix(in srgb, var(--brain) 10%, white); color: var(--ink); }
.va-chip.del { background: color-mix(in srgb, var(--fail) 10%, white); text-decoration: line-through; }
details.va-log { margin-top: 0.3rem; } details.va-log summary { color: var(--muted); cursor: pointer; font-size: 0.88em; }
details.va-log pre { font-family: 'IBM Plex Mono', monospace; font-size: 0.78em; white-space: pre-wrap;
  background: var(--paper); padding: 0.6rem; border-radius: 6px; max-height: 18rem; overflow: auto; }
.va-links { margin: 0.35rem 0 0 !important; padding: 0 !important; list-style: none !important; }
.va-links li { margin: 0.15rem 0 !important; font-size: 0.9em; overflow-wrap: anywhere; }
.va-links a, .va-src a { color: var(--web); }
.va-src { margin-top: 0.35rem; font-size: 0.88em; color: var(--muted); overflow-wrap: anywhere; }
.va-empty { color: var(--muted); padding: 2.5rem 0.5rem; text-align: center; }

/* result */
.va-result { background: var(--panel); border: 1px solid var(--line); border-top: 4px solid var(--c);
  padding: 1.1rem 1.3rem; border-radius: 10px; }
.va-result.ok { --c: var(--ok); } .va-result.fail { --c: var(--fail); }
.va-result h2 { margin: 0 0 0.2rem !important; font-size: 1.25rem !important; color: var(--ink); font-weight: 600; }
.va-result .sub { color: var(--muted); margin: 0 0 1rem !important; }
.va-result .va-metrics { display: grid !important; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)) !important;
  gap: 0.6rem !important; margin-bottom: 1.1rem; }
.va-result .va-metric { background: var(--paper); border-radius: 8px; padding: 0.6rem 0.8rem; margin: 0 !important; }
.va-metric .v { font-size: 1.3rem; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
.va-metric .k { color: var(--muted); font-size: 0.82em; }
.va-result table.va-table { border-collapse: collapse; width: 100%; max-width: 760px; margin: 0 !important; }
.va-result table.va-table, .va-result table.va-table th, .va-result table.va-table td {
  border: none !important; border-bottom: 1px solid var(--line) !important; background: transparent !important; }
.va-result table.va-table th { text-align: left; color: var(--muted); font-weight: 500; font-size: 0.85em; padding: 0.35rem 0.6rem 0.35rem 0 !important; }
.va-result table.va-table td { padding: 0.4rem 0.6rem 0.4rem 0 !important; font-family: 'IBM Plex Mono', monospace; font-size: 0.86em; }
.va-result table.va-table tr.changed td.after { color: var(--ok); font-weight: 500; }
.va-result table.va-table tr.changed td.after::after { content: "  changed"; font-family: 'IBM Plex Sans', sans-serif;
  color: var(--muted); font-weight: 400; font-size: 0.85em; }
.va-divider { display: flex; align-items: center; gap: 0.6rem; color: var(--brain); font-weight: 600;
  font-size: 0.82rem; letter-spacing: 0.03em; margin: 0.2rem 0 1rem 1.9rem; text-transform: uppercase; }
.va-divider::after { content: ""; flex: 1; height: 1px; background: color-mix(in srgb, var(--brain) 35%, transparent); }
.va-best { background: color-mix(in srgb, var(--ok) 10%, white); border-radius: 8px; padding: 0.6rem 0.9rem;
  margin: 0 0 1rem !important; color: var(--ink); }
.va-best.same { background: var(--paper); }
.va-metric .was { color: var(--muted); font-size: 0.75em; text-decoration: line-through; margin-right: 0.3rem; font-weight: 500; }
.va-metric .up { color: var(--ok); font-size: 0.7em; margin-left: 0.3rem; }
.va-opt-hint { color: var(--muted); font-size: 0.9rem; margin: 0.3rem 0 0 !important; }
.va-note { margin: 0.8rem 0 0 !important; color: var(--ink); }
.va-next { white-space: pre-wrap; line-height: 1.6; margin: 0.4rem 0 0; color: var(--ink); }
"""

STEPS = {
    "gpu": ("Detected hardware", "info"),
    "start": ("Starting point", "info"),
    "size_check": ("Size check, before downloading", "warn"),
    "running": ("Running vLLM on the GPU", "run"),
    "failed": ("vLLM failed", "fail"),
    "asking": ("Asking Nemotron", "brain"),
    "fix": ("Nemotron's fix", "brain"),
    "checking": ("Checking the web before giving up", "web"),
    "search": ("Searched the web (Tavily)", "web"),
    "give_up": ("Agent's conclusion", "fail"),
    "environment": ("Environment problem", "fail"),
    "done": ("Works", "ok"),
    "cannot_run": ("No working configuration", "fail"),
    "opt_asking": ("Asking Nemotron", "brain"),
    "candidate": ("Nemotron's faster config", "brain"),
    "cand_failed": ("That config failed", "warn"),
    "opt_skip": ("Proposal not used", "warn"),
    "opt_second": ("Second opinion", "brain"),
    "measured": ("Measured", "ok"),
    "optimized": ("Faster config found", "ok"),
    "opt_none": ("Working config kept", "info"),
}
esc = lambda x: html.escape(str(x))


def link(url, title=None):
    url = str(url or "")
    if not url.startswith(("http://", "https://")):
        return esc(title or url)
    return f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(title or url)}</a>'


def nemotron_name(e):
    return e.get("nemotron", "").split("/")[-1].replace("_", ".").replace("-", " ")


def run_status(events, busy):
    if not events:
        return "Idle", "info"
    last = events[-1]["type"]
    if last == "optimized":
        sp = events[-1].get("speedup") or 0
        return (f"Optimized · {sp}×" if sp >= 1.05 else "Optimized · more context"), "ok"
    if last in ("done", "opt_none"):
        return "Works", "ok"
    if last in ("cannot_run", "give_up", "environment"):
        return "Cannot run", "fail"
    if not busy:
        return "Stopped", "info"
    if last in ("search", "checking"):
        return "Searching the web", "web"
    if last in ("asking", "fix", "failed"):
        return "Nemotron is fixing it", "brain"
    if last in ("opt_start", "opt_asking", "candidate", "measured", "cand_failed", "opt_skip", "opt_second"):
        return "Optimizing", "brain"
    return "Running vLLM", "run"


def render_trace(events, busy, title=""):
    if not events:
        return ('<div class="va-empty">Nothing running yet.<br>Play a recorded run, or start a live run on your GPU.'
                '<br>Every step the agent takes will appear here.</div>')
    status, cls = run_status(events, busy)
    times = [e.get("time") for e in events if e.get("time")]
    meta = f"{times[0]} – {times[-1]}" if len(times) > 1 else (times[0] if times else "")
    head = (f'<div class="va-runhead"><div><div class="what">{esc(title)}</div><div class="meta">{esc(meta)}</div></div>'
            f'<span class="va-status {cls}">{esc(status)}</span></div>')
    items = []
    for i, e in enumerate(events):
        if e["type"] == "opt_start":
            items.append(f'</ol><div class="va-divider">{esc(e["text"])}</div><ol class="va-trace">')
            continue
        name, cls = STEPS.get(e["type"], (e["type"], "info"))
        if e["type"] == "measured" and not e.get("ok"):
            cls = "warn"
        active = busy and i == len(events) - 1 and e["type"] in ("running", "asking", "checking", "opt_asking")
        who = (f'<span class="who">{esc(nemotron_name(e))}</span>'
               if e["type"] in ("asking", "fix", "give_up", "opt_asking", "candidate") else "")
        text = e["text"]
        if e["type"] == "failed":
            text = clean_error(e["cause"]) if e.get("cause") else text.replace("FAILED: ", "", 1)
        elif e["type"] == "give_up":
            text = text.split(" -> ", 1)[-1]
        elif e["type"] == "cannot_run" and e.get("gpu"):
            g = e["gpu"]
            text = f"Tried on {g['name']} ({g['memory']}) with vLLM {g['vllm_version']}. Details below."
        elif e["type"] == "asking":
            text = ("Reading the web results ..." if "again" in text
                    else "Reading the root cause and this GPU's facts ...")
        elif e["type"] == "fix":
            text = e.get("reason", text)
        elif e["type"] == "opt_asking":
            text = "Reading the working run: " + e["facts"] if e.get("facts") else \
                   "Reading the measured speed, memory and KV cache of the working run ..."
        elif e["type"] == "optimized":
            text = (f"{e.get('speedup')}× the throughput of the working config, verified on this GPU."
                    if (e.get("speedup") or 0) >= 1.05 else e["text"].replace("Optimized: ", "").capitalize() + ".")
        body = f'<div class="va-text">{esc(text)}</div>'
        if e["type"] in ("fix", "candidate"):
            chips = [f'<span class="va-chip">{esc(k)}: {esc(v)}</span>' for k, v in (e.get("changes") or {}).items()]
            chips += [f'<span class="va-chip del">{esc(k)}</span>' for k in e.get("removed") or []]
            if e.get("new_model"):
                chips.insert(0, f'<span class="va-chip">model: {esc(e["new_model"])}</span>')
            if chips:
                body += f'<div class="va-chips">{"".join(chips)}</div>'
        if e["type"] == "search":
            results = e.get("results") or []
            body = f'<div class="va-text">{len(results)} relevant result(s)' + ("" if results else
                    ", so Nemotron works from the error alone") + '</div>'
            if results:
                body += '<ul class="va-links">' + "".join(f'<li>{link(r.get("url"), r.get("title") or r.get("url"))}</li>'
                                                          for r in results) + '</ul>'
        if e["type"] in ("fix", "give_up") and e.get("source"):
            body += f'<div class="va-src">Source: {link(e["source"])}</div>'
        if e["type"] in ("failed", "cand_failed") and (e.get("log") or e.get("cause")):
            detail = e.get("log") or e.get("cause")
            body += f'<details class="va-log"><summary>Show the error</summary><pre>{esc(detail[-4000:])}</pre></details>'
        items.append(f'<li class="va-step {cls}{" active" if active else ""}">'
                     f'<span class="va-title">{esc(name)}{who}</span><span class="va-time">{esc(e.get("time", ""))}</span>'
                     f'{body}</li>')
    return head + f'<ol class="va-trace">{"".join(items)}</ol>'


CELL = 'style="border:none !important; border-bottom:1px solid #DCE2E6 !important; background:transparent !important"'   # inline, so Gradio's own table styles can't add boxes


def render_result(final):
    if not final:
        return ""
    if final["type"] == "done":
        m, gpu = final["metrics"], final["gpu"]
        outcome = (f"fixed by the agent after {final['failures']} failed attempt(s)" if final.get("failures")
                   else "worked first time")
        metrics = [("Throughput, 64 users", f"{m['tokens_per_s']} tok/s"),
                   ("Latency, 1 user", f"{m['ms_per_token']} ms/token"),
                   ("Context", f"{m['context']:,} tokens" if m.get("context") else "n/a"),
                   ("GPU memory", f"{m['gpu_memory_gb']} GB"),
                   ("Accuracy check", m["accuracy"]),
                   ("Load time", f"{m['load_s']} s")]
        cards = "".join(f'<div class="va-metric" style="margin:0 !important"><div class="v">{esc(v)}</div>'
                        f'<div class="k">{esc(k)}</div></div>' for k, v in metrics)
        cards = ('<div class="va-metrics" style="display:grid !important; gap:0.6rem !important; '
                 'grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)) !important">' + cards + '</div>')
        rows = "".join(f'<tr class="{"changed" if r["changed"] else ""}"><td {CELL}>{esc(r["setting"])}</td>'
                       f'<td {CELL}>{esc(r["requested"])}</td><td class="after" {CELL}>{esc(r["delivered"])}</td></tr>'
                       for r in final["rows"])
        table = (f'<table class="va-table" style="border:none !important; border-collapse:collapse !important">'
                 f'<tr><th {CELL}>Setting</th><th {CELL}>You asked for</th><th {CELL}>Delivered</th></tr>{rows}</table>'
                 if rows else '<p class="va-note">vLLM defaults, no settings changed.</p>')
        switched = (f'<p class="va-note">Model switched to <b>{esc(final["model"])}</b> so it fits this GPU.</p>'
                    if final["model"] != final["model_id"] else "")
        return (f'<div class="va-result ok"><h2>{esc(final["model"])} runs on {esc(gpu["name"])}</h2>'
                f'<p class="sub">{esc(final["mode"].capitalize())} mode, {esc(outcome)}. Tested with vLLM '
                f'{esc(gpu["vllm_version"])}.</p>{cards}{table}{switched}{sources_html(final)}</div>')
    gpu = final["gpu"]
    return (f'<div class="va-result fail"><h2>{esc(final["model_id"])}: no working configuration found</h2>'
            f'<p class="sub">Tried on {esc(gpu["name"])} ({esc(gpu["memory"])}) with vLLM {esc(gpu["vllm_version"])}. '
            f'{esc(clean_error(final.get("reason", "")) or final.get("reason", ""))}</p>'
            + (f'<p class="va-note"><b>Agent:</b> {esc(final["advice"])}</p>' if final.get("advice") else "")
            + f'<p class="va-note"><b>What you can do next</b></p><div class="va-next">{esc(final.get("next_steps", ""))}</div>'
            + sources_html(final) + '</div>')


def render_final(events):
    """The result card for a run: working result, plus the optimization outcome if there is one."""
    done = next((e for e in reversed(events) if e["type"] in ("done", "cannot_run")), None)
    opt = next((e for e in reversed(events) if e["type"] in ("optimized", "opt_none")), None)
    if opt and opt["type"] == "optimized":
        return render_optimized(done, opt)
    card = render_result(done)
    if opt and done:
        note = (f'<p class="va-best same">⚡ <b>Optimize:</b> the working config was kept. '
                f'{esc(opt.get("text", ""))}</p>')
        card = card.replace("</p>", "</p>" + note, 1)
    return card


def render_optimized(done, opt):
    b, a, gpu = opt["before"], opt["after"], opt["gpu"]
    def card(k, new, old=None, up=None):
        was = f'<span class="was">{esc(old)}</span>' if old not in (None, new) else ""
        upx = f'<span class="up">{esc(up)}</span>' if up else ""
        return (f'<div class="va-metric" style="margin:0 !important"><div class="v">{was}{esc(new)}{upx}</div>'
                f'<div class="k">{esc(k)}</div></div>')
    ctx = lambda m: f"{m['context']:,}" if m.get("context") else "n/a"
    ctx_up = (f"{a['context'] / b['context']:.0f}×" if a.get("context") and b.get("context")
              and a["context"] >= 2 * b["context"] else None)
    cards = (card("Throughput, 64 users (tok/s)", f"{a['tokens_per_s']:,}", f"{b['tokens_per_s']:,}", f"{opt['speedup']}×")
             + card("Latency, 1 user (ms/token)", a["ms_per_token"], b["ms_per_token"])
             + card("Context (tokens)", ctx(a), ctx(b), ctx_up)
             + card("GPU memory (GB)", a["gpu_memory_gb"], b["gpu_memory_gb"])
             + card("Accuracy check", a["accuracy"], b["accuracy"])
             + card("Load time (s)", a["load_s"]))
    cards = ('<div class="va-metrics" style="display:grid !important; gap:0.6rem !important; '
             'grid-template-columns:repeat(auto-fit, minmax(135px, 1fr)) !important">' + cards + '</div>')
    green = 'style="border:none !important; border-bottom:1px solid #DCE2E6 !important; background:transparent !important; color:#1D7A5A; font-weight:600"'
    rows = "".join(f'<tr><td {CELL}>{esc(r["setting"])}</td><td {CELL}>{esc(r["requested"])}</td>'
                   f'<td {CELL}>{esc(r["working"])}</td>'
                   f'<td {green if r["optimized"] != r["working"] else CELL}>{esc(r["optimized"])}</td></tr>'
                   for r in opt["rows"])
    table = (f'<table class="va-table" style="border:none !important; border-collapse:collapse !important">'
             f'<tr><th {CELL}>Setting</th><th {CELL}>You asked for</th><th {CELL}>Working</th>'
             f'<th {CELL}>Optimized</th></tr>{rows}</table>')
    more_ctx = f" and {ctx_up} the context" if ctx_up else ""
    headline = (f'⚡ <b>{esc(opt["speedup"])}× the throughput</b>{esc(more_ctx)} of the first working config, '
                'with the same accuracy check.' if (opt.get("speedup") or 0) >= 1.05 else
                f'⚡ <b>{esc(ctx_up or "More")} context at the same speed</b> as the first working config, '
                'with the same accuracy check.')
    return (f'<div class="va-result ok"><h2>{esc(opt["model"])} runs on {esc(gpu["name"])} · optimized</h2>'
            f'<p class="sub">{esc(done["mode"].capitalize())} mode. Made to work, then optimized and verified on this GPU '
            f'with vLLM {esc(gpu["vllm_version"])}.</p>'
            f'<p class="va-best">{headline}</p>{cards}{table}'
            f'<p class="va-note" style="color:#5E6E7A">Downloads: <b>inference.py</b> (optimized), '
            f'inference_working.py, requirements.txt</p></div>')


def sources_html(final):
    srcs = final.get("sources") or []
    if not srcs:
        return ""
    return ('<p class="va-note"><b>Sources</b></p><ul class="va-links">'
            + "".join(f"<li>{link(u)}</li>" for u in srcs) + "</ul>")


def outputs_for(events, folder):
    """Code preview + download files for a finished run (the optimized script, if there is one)."""
    final = next((e for e in reversed(events) if e["type"] in ("done", "cannot_run")), None)
    opt = next((e for e in reversed(events) if e["type"] == "optimized"), None)
    if not final or final["type"] != "done":
        return gr.update(value="", visible=False), gr.update(value=None, visible=False)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "requirements.txt").write_text(final["requirements"])
    files = [folder / "inference.py"]
    if opt:
        (folder / "inference.py").write_text(opt["inference_py"])
        (folder / "inference_working.py").write_text(final["inference_py"])
        files.append(folder / "inference_working.py")
    else:
        (folder / "inference.py").write_text(final["inference_py"])
    files.append(folder / "requirements.txt")
    return (gr.update(value=(opt or final)["inference_py"], visible=True),
            gr.update(value=[str(f) for f in files], visible=True))


NO_OPT = (gr.update(visible=False), None)             # optimize button row hidden, nothing to optimize
HIDDEN = (gr.update(visible=False), gr.update(visible=False)) + NO_OPT


def save_recording(path, title, model, events):
    RECORDINGS.mkdir(exist_ok=True)
    slim = [{**e, "log": e.get("log", "")[-4000:]} if "log" in e else e for e in events]
    json.dump({"title": title, "model": model, "recorded": time.strftime("%Y-%m-%d"), "events": slim},
              open(path, "w"), indent=1)


def message(title, text):
    return f'<div class="va-result fail"><h2>{esc(title)}</h2><p class="sub">{esc(text)}</p></div>'


# ------------------------------------------------------------------ live runs
def live_run(model, settings_text, max_context, search_first=False):
    model = (model or "").strip()
    if not gpu_available():
        yield render_trace([], False), message("No GPU here", "Live runs need an NVIDIA GPU with vLLM, for example "
              "the Colab notebook in the repository. Recorded runs work without a GPU."), *HIDDEN
        return
    if not model:
        yield render_trace([], False), message("Enter a model name", "For example Qwen/Qwen2.5-7B-Instruct."), *HIDDEN
        return
    try:
        settings = json.loads(settings_text) if (settings_text or "").strip() else None
    except json.JSONDecodeError as err:
        yield render_trace([], False), message("Settings are not valid JSON", str(err)), *HIDDEN
        return

    max_context = str(max_context or "").strip().replace(",", "")
    if max_context and not max_context.isdigit():
        yield render_trace([], False), message("Context must be a number", "For example 32768, or leave it empty."), *HIDDEN
        return

    title = f"Live run · {model} · {'expert' if settings else 'beginner'} mode"
    run_id = time.strftime("%Y%m%d-%H%M%S")
    folder = RUNS / run_id
    folder.mkdir(parents=True, exist_ok=True)
    events, final = [], None
    try:
        for e in run_agent(model, settings, int(max_context) if max_context else None, out_dir=str(folder),
                           search_first=bool(search_first)):
            events.append(e)
            if e["type"] in ("done", "cannot_run"):
                final = e
            yield render_trace(events, final is None, title), render_final(events), *HIDDEN
    except Exception as err:
        events.append({"type": "environment", "text": f"{type(err).__name__}: {err}", "time": time.strftime("%H:%M:%S")})
        yield render_trace(events, False, title), "", *HIDDEN
        return

    # Keep a recording of this run so it can be replayed later (logs shortened)
    rec_title = f"{model} ({'expert' if settings else 'beginner'} mode, live)"
    rec_path = RECORDINGS / f"live_{run_id}.json"
    save_recording(rec_path, rec_title, model, events)
    works = final is not None and final["type"] == "done"
    state = {"events": events, "title": title, "folder": str(folder), "rec_path": str(rec_path),
             "rec_title": rec_title, "model": model} if works else None
    yield (render_trace(events, False, title), render_final(events), *outputs_for(events, folder),
           gr.update(visible=works), state)


def optimize_run(state):
    """After a successful live run: try a faster config on the same GPU and verify it."""
    if not state:
        yield render_trace([], False), message("Nothing to optimize", "Run a model live first."), *HIDDEN
        return
    events, title, folder = list(state["events"]), state["title"], Path(state["folder"])
    done = next(e for e in reversed(events) if e["type"] == "done")
    try:
        for e in optimize(done, out_dir=str(folder)):
            events.append(e)
            finished = e["type"] in ("optimized", "opt_none")
            yield render_trace(events, not finished, title), render_final(events), *HIDDEN
    except Exception as err:
        events.append({"type": "opt_none", "text": f"Optimization stopped ({type(err).__name__}: {err}); "
                                                   "the working config stays.", "time": time.strftime("%H:%M:%S")})
    save_recording(Path(state["rec_path"]), state["rec_title"], state["model"], events)
    yield render_trace(events, False, title), render_final(events), *outputs_for(events, folder), *NO_OPT


# ------------------------------------------------------------------ recorded runs
def recordings():
    items = []
    for f in sorted(RECORDINGS.glob("*.json")):
        try:
            data = json.load(open(f))
            items.append((data["title"], f.name))
        except Exception:
            continue
    return items


def refresh_list(current=None):
    items = recordings()
    names = [v for _, v in items]
    return gr.update(choices=items, value=current if current in names else (names[0] if names else None))


def replay(name):
    if not name:
        yield render_trace([], False), "", *HIDDEN
        return
    data = json.load(open(RECORDINGS / name))
    title = f"Recorded run · {data['title']}"
    events, final = [], None
    for e in data["events"]:
        events.append(e)
        if e["type"] in ("done", "cannot_run"):
            final = e
        busy = final is None or e["type"] not in ("done", "cannot_run", "optimized", "opt_none")
        yield render_trace(events, busy, title), render_final(events), *HIDDEN
        time.sleep(1.4 if e["type"] in ("running", "asking", "opt_asking") else 0.6)
    yield render_trace(events, False, title), render_final(events), *outputs_for(events, RUNS / f"replay_{Path(name).stem}"), *NO_OPT


# ------------------------------------------------------------------ layout
EXAMPLES = [
    ("7B model, too big for a T4", "Qwen/Qwen2.5-7B-Instruct", "", None),
    ("Config copied from an H100", "Qwen/Qwen2.5-0.5B-Instruct",
     '{"dtype": "bfloat16", "kv_cache_dtype": "fp8", "tensor_parallel_size": 2}', None),
    ("32B model on a 15 GB GPU", "Qwen/Qwen2.5-32B-Instruct", "", None),
    ("Phi-3.5 asking for 32k context", "microsoft/Phi-3.5-mini-instruct", "", 32768),
]


def app_bar():
    web = ('<span class="va-pill" style="--c:#5FC4D1"><span class="dot"></span>Tavily web search</span>'
           if os.environ.get("TAVILY_API_KEY") else "")
    gpu = (f'<span class="va-pill" style="--c:#4FB38A"><span class="dot"></span>{esc(GPU_LABEL)}</span>' if GPU_LABEL
           else '<span class="va-pill" style="--c:#C9D3DA"><span class="dot"></span>No GPU: recorded runs only</span>')
    return (f'<div class="va-bar"><div><div class="name">{APP_NAME}</div><div class="tag">{esc(TAGLINE)}</div></div>'
            f'<div class="va-pills">{gpu}<span class="va-pill" style="--c:#8EA6E8"><span class="dot"></span>'
            f'Nemotron on Nebius Token Factory</span>{web}</div></div>')


def build():
    try:
        blocks = gr.Blocks(title=APP_NAME, theme=gr.themes.Base())
    except TypeError:
        blocks = gr.Blocks(title=APP_NAME)
    with blocks as demo:
        gr.HTML(f"<style>{CSS}</style>")
        gr.HTML(app_bar())
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=320, elem_classes="va-panel"):
                with gr.Tab("Recorded runs"):
                    gr.HTML('<p class="va-hint">Real runs on a Tesla T4, replayed step by step. No GPU needed.</p>')
                    first = (recordings() or [(None, None)])[0][1]
                    with gr.Row():
                        rec = gr.Dropdown(choices=recordings(), value=first, show_label=False, scale=5)
                        refresh = gr.Button("↻", scale=0, min_width=44)
                    play = gr.Button("Play this run", variant="primary")
                with gr.Tab("Run live"):
                    gr.HTML('<p class="va-hint">' + ("Runs on this machine's GPU. Loading a model takes a few minutes."
                            if GPU_LABEL else "No GPU detected. Use the Colab notebook to run live.") + "</p>")
                    model = gr.Textbox(label="Hugging Face model", placeholder="Qwen/Qwen2.5-7B-Instruct")
                    with gr.Accordion("Expert settings (optional)", open=False):
                        settings = gr.Textbox(label="Your vLLM settings (JSON)", lines=2,
                                              placeholder='{"kv_cache_dtype": "fp8", "max_num_seqs": 128}',
                                              info="If yours fail, the agent changes only what is broken.")
                        max_ctx = gr.Textbox(label="Context to try first (tokens)", placeholder="Optional, e.g. 32768")
                        search_first = gr.Checkbox(label="Search the web on the first failure",
                                                   info="Default: only when Nemotron's first fix doesn't work, "
                                                        "or before it gives up. Needs TAVILY_API_KEY.", value=False)
                    run = gr.Button("Run the agent", variant="primary")
                    gr.HTML('<p class="va-hint" style="margin-top:0.8rem">Or start from a scenario:</p>')
                    ex_buttons = []
                    with gr.Column(elem_classes="va-examples"):
                        for label, *_ in EXAMPLES:
                            ex_buttons.append(gr.Button(label, size="sm", variant="secondary"))
            with gr.Column(scale=7, elem_classes="va-panel"):
                trace = gr.HTML(render_trace([], False))
        result = gr.HTML()
        with gr.Row(visible=False, equal_height=True) as opt_row:
            opt_btn = gr.Button("⚡ Optimize for this GPU", variant="primary", scale=0, min_width=240)
            gr.HTML('<p class="va-opt-hint">Nemotron reads the measured numbers of this run, proposes a faster '
                    'config, and the agent verifies it on this GPU. Takes a few minutes.</p>')
        last = gr.State(None)
        with gr.Row(equal_height=False):
            code = gr.Code(label="inference.py (tested on this GPU)", language="python", visible=False, scale=3)
            files = gr.File(label="Download", file_count="multiple", visible=False, scale=1)

        outs = [trace, result, code, files, opt_row, last]
        play.click(replay, inputs=rec, outputs=outs)
        refresh.click(refresh_list, inputs=rec, outputs=rec)
        run.click(live_run, inputs=[model, settings, max_ctx, search_first], outputs=outs,
                  concurrency_limit=1, concurrency_id="gpu").then(refresh_list, inputs=rec, outputs=rec)
        opt_btn.click(optimize_run, inputs=last, outputs=outs,
                      concurrency_limit=1, concurrency_id="gpu").then(refresh_list, inputs=rec, outputs=rec)
        for btn, (_, m, s, c) in zip(ex_buttons, EXAMPLES):
            btn.click(lambda m=m, s=s, c=c: (m, s, str(c) if c else ""), outputs=[model, settings, max_ctx])
    return demo


demo = build()

if __name__ == "__main__":
    demo.queue().launch(share=os.environ.get("SHARE", "0") == "1")
