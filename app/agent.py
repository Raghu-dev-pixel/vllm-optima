"""
vLLM Optima: give a model, get a working inference script.

Same logic as the tested notebook (vllm_agent_Final1.ipynb), packaged so an app can call it:

    for event in run_agent("Qwen/Qwen2.5-7B-Instruct"):
        print(event["text"])

Each step is reported as an event (a dict with "type" and "text"), so a Gradio app can show
the agent's work live: observed -> diagnosed -> changed -> verified.

Setup (once):
    - vLLM installed in its own environment; set VLLM_PYTHON to that python
      (default: /content/vllm-env/bin/python, the Colab setup)
    - NEBIUS_API_KEY environment variable (Nebius Token Factory)
    - TAVILY_API_KEY environment variable (optional: web search for unknown errors)
    - test_run.py in the same folder as this file
"""
import os, re, json, time, subprocess

from openai import OpenAI
from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("VLLM_PYTHON", "/content/vllm-env/bin/python")
TEST_RUN = os.path.join(HERE, "test_run.py")
LESSONS_FILE = os.environ.get("LESSONS_FILE", "lessons.json")

FAST_MODEL = "nvidia/Nemotron-3_5-Lightning"        # 1st fix attempt: fast and cheap, handles routine errors
STRONG_MODEL = "nvidia/Nemotron-3-Ultra-550b-a55b"  # later attempts: strongest reasoning, for harder cases
MAX_ATTEMPTS = 4
SEARCH_FROM_ATTEMPT = 2   # Nemotron first; search the web only if its fix fails (or before giving up)

_client = None
def client():
    """Nebius Token Factory client (created on first use)."""
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/",
                         api_key=os.environ.get("NEBIUS_API_KEY"))
    return _client

_tavily = None
def tavily():
    """Tavily client, or None when TAVILY_API_KEY is not set (web search then simply stays off)."""
    global _tavily
    if _tavily is None and os.environ.get("TAVILY_API_KEY"):
        try:
            from tavily import TavilyClient
            _tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        except Exception:      # tavily-python not installed: run without web search
            return None
    return _tavily


# What the agent knows (its "skill file"). Add a line for every new error you meet.
KNOWN_FIXES = """
- GPU compute capability below 8.0 (e.g. T4): no bfloat16 -> dtype "half".
- "FP8 KV cache is not supported" -> remove kv_cache_dtype or use "auto".
- kv_cache_dtype accepts only values from the error list (e.g. "auto", "fp8").
- "Unknown quantization method: X. Must be one of [...]" -> choose only from that list.
- "ModelTooLargeError" / model too big for GPU memory -> switch "model" to a pre-quantized
  variant of the SAME model that exists on Hugging Face, e.g. "Qwen/Qwen2.5-7B-Instruct-AWQ"
  (4-bit, about 1/3 of the size). If even the 4-bit size is above about 85% of the GPU memory,
  set "give_up": true and say which GPU size would be needed.
- "ModelNotFoundError" -> that model name doesn't exist; use a real variant name or give up.
- "KV cache ... max seq len" / "No available memory for the cache blocks" -> lower max_model_len
  (not below the minimum) or raise gpu_memory_utilization (max 0.9).
- Out of memory during CUDA graph capture -> lower max_num_seqs or set enforce_eager true.
- "contains custom code which must be executed" -> the model needs trust_remote_code, which the agent
  must NEVER enable itself (it runs code from the internet). Set "give_up": true and tell the user they
  can add {"trust_remote_code": true} as their own expert setting if they trust the repository.
"""
ALLOWED = ["dtype", "max_model_len", "gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens",
           "enforce_eager", "enable_prefix_caching", "enable_chunked_prefill", "kv_cache_dtype",
           "quantization", "tensor_parallel_size", "cpu_offload_gb"]
HARMLESS = ("FA version 2", "FA2 is only supported", "FlashInfer", "unauthenticated requests")
ENVIRONMENT = {"compiled with different CUDA versions": "library CUDA mismatch: rebuild vllm-env",
               "ModuleNotFoundError": "missing package: install it in vllm-env",
               "No CUDA GPUs are available": "no GPU: Runtime > Change runtime type > GPU"}


# ---------------------------------------------------------------- helpers (unchanged logic)
def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

def detect_gpu():
    """GPU name, memory, compute capability, count and vLLM version (measured, never guessed)."""
    gpus = sh(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"]).splitlines()
    if not gpus:
        raise RuntimeError("No NVIDIA GPU visible.")
    name, memory, compute_cap = [x.strip() for x in gpus[0].split(",")]
    return {"name": name, "memory": memory, "compute_capability": compute_cap, "count": len(gpus),
            "vllm_version": sh([PY, "-c", "import vllm; print(vllm.__version__)"])}

def load_lessons():
    return json.load(open(LESSONS_FILE)) if os.path.exists(LESSONS_FILE) else []

def root_cause(log):
    """The first real 'SomethingError: ...' line, with the lines around it."""
    lines = [l for l in log.splitlines() if not any(h in l for h in HARMLESS)]
    for i, line in enumerate(lines):
        if re.search(r"\b\w+(Error|Exception): ", line):
            return "\n".join(lines[max(0, i - 2):i + 4])
    return "\n".join(lines[-15:])

def summary(cause):
    """One line: the error line plus the next line (some messages continue there)."""
    lines = cause.splitlines()
    i = next((i for i, l in enumerate(lines) if re.search(r"\b\w+(Error|Exception): ", l)), -1)
    return " ".join(l.strip() for l in lines[i:i + 2])[:200]

# Errors the skill file already covers: no web search needed for these
KNOWN_ERRORS = ["ModelTooLargeError", "ModelNotFoundError", "FP8 KV cache", "Bfloat16 is only supported",
                "World size", "KV cache is needed", "No available memory for the cache blocks",
                "Unknown quantization method", "CUDA out of memory"]

def clean_error(cause):
    """The error summary without log prefixes like '(EngineCore pid=123) ERROR 10-07 ... [core.py:1433]'."""
    s = summary(cause)
    s = re.sub(r"\(EngineCore pid=\d+\)\s*", "", s)
    s = re.sub(r"ERROR \d\d-\d\d \S+ \[[^\]]+\]\s*", "", s)
    return s.replace("pydantic_core._pydantic_core.", "").strip()

def same_error(a, b):
    """Same error, ignoring log prefixes (process id, time) that differ between runs."""
    a, b = clean_error(a), clean_error(b)
    return bool(a and b) and (a.startswith(b) or b.startswith(a))

def is_known(cause, lessons):
    """Known = covered by the skill file or by a lesson from an earlier run."""
    text = summary(cause)
    return any(k in text for k in KNOWN_ERRORS) or any(same_error(l["error"], cause) for l in lessons)

_search_cache = {}
def search_web(cause):
    """One Tavily search for an unknown error. Returns only results that are actually
    about the error (they must mention the setting/name in quotes), or [] if none are."""
    if tavily() is None:
        return []
    err = clean_error(cause)
    names = re.findall(r"'([A-Za-z_][\w.\-/]*)'", err)                  # e.g. 'max_seq_len_to_capture'
    query = (f"vLLM {names[0]} removed deprecated replacement" if names else f"vLLM {err}")[:380]
    if query in _search_cache:
        return _search_cache[query]
    try:
        found = tavily().search(query=query, max_results=5, search_depth="advanced",
                                include_domains=["docs.vllm.ai", "github.com", "huggingface.co", "discuss.vllm.ai"])
        results = [{"title": r["title"], "url": r["url"], "content": r["content"][:500]}
                   for r in found.get("results", [])]
    except Exception:
        results = []
    if names:   # keep only results that mention the name from the error
        results = [r for r in results if any(n.lower() in (r["title"] + r["content"]).lower() for n in names)]
    results = results[:3]
    _search_cache[query] = results
    return results

def web_block(web):
    return "".join(f"\n- {w['title']} ({w['url']}): {w['content']}" for w in web)

def model_size_gb(model):
    """Size of the model's weight files on Hugging Face, WITHOUT downloading them.
    Returns None if unknown, or the string "missing" if the model doesn't exist."""
    try:
        files = HfApi().model_info(model, files_metadata=True).siblings
    except Exception as e:
        return "missing" if "NotFound" in type(e).__name__ else None
    weights = [f for f in files if f.rfilename.endswith(".safetensors")] or \
              [f for f in files if f.rfilename.endswith(".bin")]
    return sum(f.size or 0 for f in weights) / 1e9 or None

def run_steps(model, settings, gpu_gb):
    """Check, then run test_run.py on this GPU. Yields ("size_check" | "running", text) progress
    steps AS THEY HAPPEN, and finally ("result", result) with the results or the root cause."""
    # Quick check first: a model that can't fit fails instantly, before any download
    size = model_size_gb(model)
    if size == "missing":
        yield "result", {"ok": False, "environment": None,
                         "cause": f"ModelNotFoundError: {model} does not exist on Hugging Face."}
        return
    if size and size > 0.85 * gpu_gb and not settings.get("cpu_offload_gb"):
        yield "size_check", f"{model}: weights {size:.1f} GB > GPU {gpu_gb:.1f} GB -> skipped the download"
        yield "result", {"ok": False, "environment": None,
                         "cause": f"ModelTooLargeError: the weights of {model} are {size:.1f} GB, but the "
                                  f"GPU has {gpu_gb:.1f} GB in total (weights must leave room for the KV cache)."}
        return
    # Announce the run BEFORE it starts (it can take minutes), so the app shows it live
    yield "running", (f"running {model} ({size:.1f} GB) with {settings or 'vLLM defaults'} ..." if size
                      else f"running {model} with {settings or 'vLLM defaults'} ...")
    try:
        p = subprocess.run([PY, TEST_RUN, model, json.dumps(settings)],
                           capture_output=True, text=True, timeout=1800)
        log = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        log = "TimeoutError: no result after 30 minutes"
    subprocess.run(["pkill", "-f", "test_run.py"])      # free the GPU
    open("vllm.log", "w").write(log)
    result = [l for l in log.splitlines() if l.startswith("RESULT:")]
    if result:
        yield "result", {"ok": True, **json.loads(result[-1][7:])}
        return
    env = next((fix for text, fix in ENVIRONMENT.items() if text in log), None)
    yield "result", {"ok": False, "cause": root_cause(log), "environment": env, "log": log}

def ask_nemotron(model, history, gpu, model_id, start, min_context, lessons):
    """Ask for a fixed configuration. Returns a validated dict, retrying on bad replies."""
    rules = f"""You fix vLLM configurations so they run on this GPU: {json.dumps(gpu)}.
Reply with ONLY JSON: {{"model": "...", "settings": {{...}}, "reason": "one sentence", "give_up": false, "source": ""}}
- If web search results are given: cite one in "source" ONLY if it directly explains this error
  or its fix (e.g. the setting was removed or renamed). If none do, leave "source" empty.
- Change as FEW settings as possible; keep everything that did not cause the failure.
- Only these settings: {ALLOWED + [k for k in start if k not in ALLOWED]}
- max_model_len must stay at least {min_context} if set.
- Keep model "{model_id}" unless it cannot fit; then only a quantized variant of the same model.
- If nothing can work on this GPU, set "give_up": true and explain.
Known fixes:{KNOWN_FIXES}
Lessons from earlier runs: {json.dumps(lessons[-10:])}"""
    for _ in range(3):
        try:
            r = client().chat.completions.create(model=model, temperature=0, max_tokens=4096,
                    messages=[{"role": "system", "content": rules}] + history)
            text = (r.choices[0].message.content or "") if r.choices else ""
            fix = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        except Exception:
            time.sleep(10)                    # busy / rate-limited / bad JSON: wait and retry
            continue
        # Allowed: the standard list + any setting the user chose themselves (never silently dropped)
        fix["settings"] = {k: v for k, v in fix.get("settings", {}).items() if k in ALLOWED or k in start}
        fix["model"] = fix.get("model") or model_id
        context_ok = int(fix["settings"].get("max_model_len", min_context)) >= min_context
        model_ok = model_id.split("/")[-1].lower() in fix["model"].lower()
        if context_ok and model_ok:
            return fix
        history.append({"role": "user", "content": "Rejected: keep max_model_len >= "
                        f"{min_context} and use only {model_id} or its quantized variant."})
    return {"give_up": True, "reason": "Nemotron did not return a usable answer"}

def inference_script(gpu, model, settings):
    return f'''# Tested by the agent on: {gpu['name']} x{gpu['count']}, vLLM {gpu['vllm_version']}
# Use the same GPU type and vLLM version, or run the agent again on your GPU.
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(model={model!r}{(', **' + repr(settings)) if settings else ''})
    out = llm.chat([{{"role": "user", "content": "Hello! Who are you?"}}],
                   SamplingParams(temperature=0.7, max_tokens=128))
    print(out[0].outputs[0].text)
'''


# ---------------------------------------------------------------- the agent
def event(kind, text, **data):
    return {"type": kind, "text": text, "time": time.strftime("%H:%M:%S"), **data}

def run_agent(model_id, user_settings=None, max_context=None, out_dir=".", search_first=False):
    """Run the agent and yield one event per step. The last event is "done" or "cannot_run".
    search_first=True searches the web on the first failure (instead of only after a failed fix)."""
    search_from = 1 if search_first else SEARCH_FROM_ATTEMPT
    # STEP 3 - Detect the hardware and pick smart starting settings
    gpu = detect_gpu()
    yield event("gpu", f"Detected {gpu['name']} x{gpu['count']} ({gpu['memory']}, compute "
                       f"{gpu['compute_capability']}), vLLM {gpu['vllm_version']}", gpu=gpu)
    gpu_gb = float(gpu["memory"].split()[0]) / 1024 * gpu["count"]

    smart = {}
    if float(gpu["compute_capability"]) < 8.0:
        smart["dtype"] = "half"                     # older GPUs (e.g. T4) have no bfloat16
    if gpu["count"] > 1:
        smart["tensor_parallel_size"] = gpu["count"]  # split the model across all GPUs
    if max_context:
        smart["max_model_len"] = max_context

    mode = "expert" if user_settings else "beginner"
    start = dict(user_settings) if user_settings else smart
    min_context = min(4096, int(start.get("max_model_len", 4096)))   # fixes may not go below this
    yield event("start", f"Mode: {mode} | starting settings: {start or 'vLLM defaults'}",
                mode=mode, start=start)

    lessons = load_lessons()

    # STEP 5 - The agent: run -> if it fails, Nemotron fixes it -> run again
    model, settings = model_id, start
    for kind, payload in run_steps(model, settings, gpu_gb):
        if kind == "result":
            result = payload
        else:
            yield event(kind, payload)
    history, failures, advice = [], [], None
    sources, last_web = [], []

    def web_event(web):
        return event("search", f"Searched the web (Tavily): {len(web)} relevant result(s)",
                     results=[{"title": w["title"], "url": w["url"]} for w in web])

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if result["ok"]:
            break
        if result["environment"]:
            advice = result["environment"]
            yield event("environment", f"Environment problem (not a settings issue): {advice}")
            break
        yield event("failed", f"FAILED: {summary(result['cause'])}", cause=result["cause"],
                    log=result.get("log", ""))
        failures.append(result["cause"])

        unknown = not is_known(result["cause"], lessons) and tavily() is not None
        web = []
        if unknown and attempt >= search_from:
            web = search_web(result["cause"])
            last_web = web or last_web
            yield web_event(web)
        history.append({"role": "user", "content": f"Model: {model}\nSettings: {settings}\n"
                                                   f"Failed with:\n{result['cause']}\n"
                                                   + (f"\nWeb search results about this error:{web_block(web)}\n" if web else "")
                                                   + "Return a fix."})
        nemotron = FAST_MODEL if attempt == 1 else STRONG_MODEL
        yield event("asking", f"Asking {nemotron.split('/')[-1]} for a fix ...", nemotron=nemotron)
        fix = ask_nemotron(nemotron, history, gpu, model_id, start, min_context, lessons)
        if not fix.get("give_up") and (fix["model"], fix["settings"]) == (model, settings):
            fix = {**fix, "give_up": True}          # a "fix" that changes nothing would just fail again

        # Before accepting "give up" on an unknown error, check the web once (if not done yet)
        if fix.get("give_up") and unknown and not web:
            yield event("checking", "Nemotron wants to give up; checking the web first ...")
            web = search_web(result["cause"])
            last_web = web or last_web
            yield web_event(web)
            if web:
                history.append({"role": "assistant", "content": json.dumps(fix)})
                history.append({"role": "user", "content": "Before giving up, consider these web results:"
                                f"{web_block(web)}\nReturn a fix if one of them applies; otherwise give_up "
                                "and put the most relevant URL in \"source\"."})
                nemotron = STRONG_MODEL
                yield event("asking", f"Asking {nemotron.split('/')[-1]} again, with the web results ...",
                            nemotron=nemotron)
                fix = ask_nemotron(nemotron, history, gpu, model_id, start, min_context, lessons)
                if not fix.get("give_up") and (fix["model"], fix["settings"]) == (model, settings):
                    fix = {**fix, "give_up": True}

        source = fix.get("source") or ""
        if source and source not in sources:
            sources.append(source)
        if fix.get("give_up"):
            advice = fix.get("reason")
            yield event("give_up", f"No working configuration found -> {advice}", nemotron=nemotron,
                        source=source)
            break
        changes = {k: v for k, v in fix["settings"].items() if settings.get(k) != v}
        removed = [k for k in settings if k not in fix["settings"]]
        yield event("fix", f"Fix {attempt}: {fix['reason']}", nemotron=nemotron, reason=fix["reason"], source=source,
                    changes=changes, removed=removed,
                    new_model=fix["model"] if fix["model"] != model else None)
        history.append({"role": "assistant", "content": json.dumps(fix)})
        model, settings = fix["model"], fix["settings"]
        for kind, payload in run_steps(model, settings, gpu_gb):
            if kind == "result":
                result = payload
            else:
                yield event(kind, payload)

    # STEP 6 - Results and ready-to-use files
    if result["ok"]:
        rows = []
        for k in sorted(set(start) | set(settings)):
            before, after = start.get(k, "(default)"), settings.get(k, "(default)")
            rows.append({"setting": k, "requested": before, "delivered": after, "changed": before != after})

        script = inference_script(gpu, model, settings)
        requirements = f"vllm=={gpu['vllm_version']}\n"
        open(os.path.join(out_dir, "inference.py"), "w").write(script)
        open(os.path.join(out_dir, "requirements.txt"), "w").write(requirements)

        for cause in failures:                      # remember what fixed each error (once)
            lesson = {"error": clean_error(cause), "fix": {"model": model, **settings}}
            if sources:
                lesson["source"] = sources[-1]
            if lesson not in lessons:
                lessons.append(lesson)
        json.dump(lessons, open(LESSONS_FILE, "w"), indent=1)

        outcome = f"fixed after {len(failures)} failure(s)" if failures else "worked first time"
        metrics = {k: result.get(k) for k in
                   ("tokens_per_s", "ms_per_token", "gpu_memory_gb", "load_s", "accuracy", "context")}
        yield event("done", f"WORKS on {gpu['name']} ({mode} mode, {outcome})", ok=True, gpu=gpu,
                    mode=mode, model_id=model_id, model=model, settings=settings, start=start,
                    metrics=metrics, rows=rows, failures=len(failures), lessons=len(lessons), sources=sources,
                    engine=result.get("engine") or {}, causes=[clean_error(c) for c in failures],
                    inference_py=script, requirements=requirements)
    else:
        reason = summary(result.get("cause", "")) or "unknown"
        size = model_size_gb(model_id)
        facts = (f"Measured model weights: {size:.1f} GB. " if isinstance(size, float) else "") + \
                f"GPU memory: {gpu['memory']}. vLLM version: {gpu['vllm_version']}."
        try:
            r = client().chat.completions.create(model=STRONG_MODEL, temperature=0, max_tokens=2048,
                messages=[{"role": "user", "content":
                    f"A user tried to run {model_id} with vLLM on {gpu['name']}. It cannot work.\n"
                    f"Facts (measured, trust these): {facts}\n"
                    f"Root cause: {reason}\nAgent's conclusion: {advice}\n"
                    + (f"Web search results about this error:{web_block(last_web)}\n" if last_web else "") +
                    "Give the user 3 short, concrete next steps, one line each, plain text, that fit THIS problem. "
                    "Only mention GPU memory or smaller models if the problem is about memory or model size. "
                    "Use the measured facts; never invent sizes, versions, flags or environment variables. "
                    "If a setting is needed that the user can add themselves, say so (e.g. as an expert setting). "
                    "If a web result explains the problem, base a step on it and include its link. No JSON."}])
            next_steps = (r.choices[0].message.content or "").strip()
        except Exception:
            next_steps = "Use a GPU with more memory, or a smaller / 4-bit version of this model."
        yield event("cannot_run", f"CANNOT RUN {model_id} on {gpu['name']} ({gpu['memory']}, x{gpu['count']})",
                    ok=False, gpu=gpu, mode=mode, model_id=model_id, reason=reason, advice=advice, sources=sources,
                    next_steps=next_steps)


# ---------------------------------------------------------------- optimize a working config
def _correct(acc):
    try:
        return int(str(acc).split("/")[0])
    except Exception:
        return 0

def is_better(new, old):
    """Faster at 64 users (or more context at the same speed), without hurting 1-user latency or accuracy."""
    if _correct(new.get("accuracy")) < _correct(old.get("accuracy")):
        return False
    if (new.get("ms_per_token") or 0) > 1.2 * (old.get("ms_per_token") or 0):
        return False
    faster = (new.get("tokens_per_s") or 0) >= 1.05 * (old.get("tokens_per_s") or 0)
    more_context = (new.get("context") or 0) > (old.get("context") or 0) and \
                   (new.get("tokens_per_s") or 0) >= 0.97 * (old.get("tokens_per_s") or 0)
    return faster or more_context

ASK_OPT = "Propose the faster configuration now. Reply with ONLY the JSON object."

def first_json(text):
    """The answer JSON in a reply. Some models write out their reasoning first (quoting settings as JSON),
    so take the LAST object that looks like an answer; a bare settings object only counts if it is the
    whole reply."""
    dec, answer = json.JSONDecoder(), None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and any(k in obj for k in ("changes", "already_best", "settings")):
            answer = obj
    if answer is None:
        try:
            whole = json.loads(text.strip().strip("`").removeprefix("json").strip())
            answer = whole if isinstance(whole, dict) else None
        except ValueError:
            pass
    return answer

def ask_optimizer(nemotron, rules, history):
    """Returns (proposal, None) or (None, why it couldn't be used)."""
    why = "no reply"
    for _ in range(2):
        text = ""
        try:
            r = client().chat.completions.create(model=nemotron, temperature=0, max_tokens=16384,
                    messages=[{"role": "system", "content": rules}] + (history or [{"role": "user", "content": ASK_OPT}]))
            text = (r.choices[0].message.content or "") if r.choices else ""
            cut = bool(r.choices) and getattr(r.choices[0], "finish_reason", None) == "length"
            found = first_json(text)
            if found is None:
                why = ("the reply was cut off before the answer" if cut else
                       f"no answer in the reply ({text.strip()[:80] or 'empty reply'})")
            else:
                t = text.strip()                           # start and end of the reply, kept in the recording
                found["_raw"] = t if len(t) <= 1200 else t[:500] + " [...] " + t[-700:]
                return found, None
        except Exception as e:
            why = f"{type(e).__name__}: {str(e)[:80]}"
        time.sleep(10)
    return None, why

# Values that are vLLM's defaults anyway: listing them changes nothing
VLLM_DEFAULTS = {"enforce_eager": False, "cpu_offload_gb": 0, "tensor_parallel_size": 1, "kv_cache_dtype": "auto",
                 "quantization": None, "dtype": "auto"}

def optimize(done, out_dir="."):
    """Try to make a WORKING config faster on this GPU. Uses the measured numbers from the working run,
    asks Lightning for ONE better config (Ultra only if that one fails or isn't faster), and verifies
    each candidate with the same benchmark. Yields events; the last one is "optimized" or "opt_none"."""
    gpu, model, start = done["gpu"], done["model"], done.get("start") or {}
    working, base = dict(done.get("settings") or {}), done["metrics"]
    gpu_gb = float(gpu["memory"].split()[0]) / 1024 * gpu["count"]
    context = int(base.get("context") or 0)
    allowed = ALLOWED + [k for k in working if k not in ALLOWED]
    yield event("opt_start", "Optimizing for this GPU")

    rules = f"""You tune a WORKING vLLM configuration so it runs faster on this GPU: {json.dumps(gpu)}.
Model: {model} (keep it). Working settings: {json.dumps(working)}
Measured on this GPU with those settings (trust these): {json.dumps(base)}
Values vLLM actually used: {json.dumps(done.get("engine") or {})}
The user originally asked for {json.dumps(start)}; errors met on the way: {json.dumps(done.get("causes") or [])}
How speed is measured: throughput = 64 chat requests at once, each a short prompt (~30 tokens) plus up to
128 new tokens, so ~160 tokens per request; latency = 1 request alone.
How vLLM uses memory: the KV cache is paged, so a request only uses KV tokens for the tokens it actually has,
NOT max_model_len. 64 benchmark requests need only ~64 x 160 = ~10,000 KV tokens. max_num_seqs below 64 directly
limits the benchmark (vLLM's default is 256). One request can use up to max_model_len tokens, so max_model_len
can be raised up to about the measured KV cache size (keep ~3% margin).
Goal: more throughput with 64 users, without making 1-user latency more than 20% worse and without
losing accuracy. A bigger context is a bonus if the KV cache has room.
Rules:
- Base every number on the measured values above (e.g. how many requests fit in the KV cache); never guess.
- List ONLY the settings you change (use null to remove one), at most 3. Don't add settings that just
  repeat vLLM's defaults (e.g. enforce_eager false, cpu_offload_gb 0, quantization null).
- If an earlier fix set things BELOW vLLM's defaults (max_num_seqs 256, gpu_memory_utilization 0.9,
  max_model_len = the model's maximum), moving back towards the defaults is usually the fastest safe choice,
  as long as you don't return to the exact values that failed.
- Never set enforce_eager true: it turns off CUDA graphs and makes decoding much slower.
- max_num_batched_tokens and max_num_seqs also use GPU memory. If the KV cache is nearly full (context
  close to the measured KV cache size), don't raise them.
- Never set max_model_len below {context}. Only these settings: {allowed}.
- max_num_batched_tokens must be at least max_model_len unless chunked prefill is on (vLLM's default).
- Don't go back to what already failed (see the errors above).
- If the working settings are already the fastest you can justify, set "already_best": true.
Known facts:{KNOWN_FIXES}
Reply with ONLY JSON: {{"changes": {{"setting": value}}, "reason": "one sentence with the numbers you used", "already_best": false}}"""
    eng = done.get("engine") or {}
    facts = (f"{base.get('tokens_per_s')} tok/s, {base.get('ms_per_token')} ms/token, context {context:,} tokens, "
             + (f"KV cache {eng['kv_cache_tokens']:,} tokens" if eng.get("kv_cache_tokens") else "KV cache size not reported")
             + (f", max_num_seqs {eng['max_num_seqs']}" if eng.get("max_num_seqs") else ""))
    history, tried, ran_ok = [], [], False
    for nemotron in (FAST_MODEL, STRONG_MODEL):
        name = nemotron.split("/")[-1]
        yield event("opt_asking", f"Asking {name} for a faster config ...", nemotron=nemotron, facts=facts)
        cand, why = ask_optimizer(nemotron, rules, history)
        if cand is None:
            yield event("opt_skip", f"{name}'s answer could not be used: {why}.")
            continue
        raw = cand.pop("_raw", "")
        changes_in = cand.get("changes")
        if changes_in is None and isinstance(cand.get("settings"), dict):      # full list given anyway
            changes_in = {k: v for k, v in cand["settings"].items() if working.get(k) != v}
        if changes_in is None and "already_best" not in cand:                  # flat answer: settings at top level
            changes_in = {k: v for k, v in cand.items() if k in allowed}
        settings = dict(working)
        for k, v in (changes_in or {}).items():
            if k not in allowed or (k not in working and VLLM_DEFAULTS.get(k, "?") == v):
                continue                                      # unknown setting, or just vLLM's default again
            if k == "enforce_eager" and v and not working.get("enforce_eager"):
                continue                                      # never faster: it turns off CUDA graphs
            if v is None:
                settings.pop(k, None)
            else:
                settings[k] = v
        if cand.get("already_best") is True:
            if nemotron == FAST_MODEL:                    # a "no gain" verdict gets a second opinion (no GPU run)
                yield event("opt_second", f"{name} thinks the working config is already the fastest"
                            + (f": {cand['reason']}" if cand.get("reason") else "") + ". Asking Ultra to double-check.",
                            nemotron=nemotron, raw=raw)
                continue
            yield event("opt_none", cand.get("reason") or "The working settings are already the fastest found.",
                        nemotron=nemotron, raw=raw)
            return
        if settings == working:
            yield event("opt_skip", f"{name} proposed no change and no reason.", nemotron=nemotron, raw=raw)
            continue
        ctx_new = int(settings.get("max_model_len") or context or 0)
        problem = None
        if settings.get("max_model_len") and int(settings["max_model_len"]) < context:
            problem = f"it would cut the context below {context:,} tokens"
        elif settings in tried:
            problem = "it repeats a config that was already tried"
        elif (settings.get("max_num_batched_tokens") and settings.get("enable_chunked_prefill") is False
              and int(settings["max_num_batched_tokens"]) < ctx_new):
            problem = (f"max_num_batched_tokens {settings['max_num_batched_tokens']} is below the context "
                       f"{ctx_new:,} with chunked prefill off, which vLLM rejects")
        if problem:
            yield event("opt_skip", f"{name}'s config was rejected before running: {problem}.", nemotron=nemotron)
            history += [{"role": "user", "content": ASK_OPT}, {"role": "assistant", "content": json.dumps(cand)},
                        {"role": "user", "content": f"Rejected: {problem}. Try once more, or already_best."}]
            continue
        tried.append(settings)
        changes = {k: v for k, v in settings.items() if working.get(k) != v}
        removed = [k for k in working if k not in settings]
        yield event("candidate", cand.get("reason", ""), nemotron=nemotron, reason=cand.get("reason", ""),
                    changes=changes, removed=removed, raw=raw)
        history += [{"role": "user", "content": ASK_OPT}, {"role": "assistant", "content": json.dumps(cand)}]
        result = None
        for kind, payload in run_steps(model, settings, gpu_gb):
            if kind == "result":
                result = payload
            else:
                yield event(kind, payload)
        ran_ok = ran_ok or result["ok"]
        if not result["ok"]:
            yield event("cand_failed", clean_error(result["cause"]), cause=result["cause"], log=result.get("log", ""))
            history.append({"role": "user", "content": f"That failed: {clean_error(result['cause'])}. "
                                                       "Propose a safer faster config, or already_best."})
            continue
        new = {k: result.get(k) for k in ("tokens_per_s", "ms_per_token", "gpu_memory_gb", "load_s",
                                          "accuracy", "context")}
        better = is_better(new, base)
        yield event("measured", f"{new['tokens_per_s']:,} tok/s · {new['ms_per_token']} ms/token · "
                                f"accuracy {new['accuracy']}" + ("" if better else " · not better"),
                    ok=better, metrics=new)
        if better:
            rows = []
            for k in sorted(set(start) | set(working) | set(settings)):
                rows.append({"setting": k, "requested": start.get(k, "(default)"),
                             "working": working.get(k, "(default)"), "optimized": settings.get(k, "(default)")})
            script = inference_script(gpu, model, settings)
            open(os.path.join(out_dir, "inference.py"), "w").write(script)
            open(os.path.join(out_dir, "inference_working.py"), "w").write(done["inference_py"])
            speedup = round((new["tokens_per_s"] or 0) / max(base.get("tokens_per_s") or 1, 1), 1)
            gain = (f"{speedup}x the throughput" if speedup >= 1.05 else
                    f"same speed, {new['context']:,} tokens of context instead of {base.get('context') or 0:,}")
            yield event("optimized", f"Optimized: {gain}" + (" of the working config" if speedup >= 1.05 else ""),
                        gain=gain,
                        gpu=gpu, model=model, before=base, after=new, rows=rows, speedup=speedup,
                        settings=settings, inference_py=script, working_py=done["inference_py"],
                        requirements=done["requirements"])
            return
        history.append({"role": "user", "content": f"Measured: {json.dumps(new)}. Not better than the working "
                                                   f"config {json.dumps(base)}. Try once more, or already_best."})
    yield event("opt_none", "None of the tested configs was faster, so the working config stays." if ran_ok else
                "No proposed config ran successfully, so the working config stays.")


# ---------------------------------------------------------------- command line / notebook use
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="vLLM Optima: give a model, get a working inference script")
    ap.add_argument("model")
    ap.add_argument("--settings", default=None, help='expert mode, e.g. \'{"kv_cache_dtype": "fp8"}\'')
    ap.add_argument("--max-context", type=int, default=None)
    ap.add_argument("--search-first", action="store_true", help="search the web on the first failure")
    a = ap.parse_args()
    for e in run_agent(a.model, json.loads(a.settings) if a.settings else None, a.max_context,
                       search_first=a.search_first):
        print(f"[{e['time']}] {e['text']}")
        for w in e.get("results", []):
            print(f"    - {w['title']}  ({w['url']})")
        if e.get("source"):
            print(f"    source: {e['source']}")
        if e["type"] == "done":
            m = e["metrics"]
            print(f"  Speed: {m['tokens_per_s']} tokens/s (64 users) | {m['ms_per_token']} ms/token (1 user)")
            print(f"  Memory: {m['gpu_memory_gb']} GB | load {m['load_s']} s | accuracy {m['accuracy']} | "
                  f"context {m['context']}")
        if e["type"] == "cannot_run":
            print("  What you can do next:\n" + e["next_steps"])
