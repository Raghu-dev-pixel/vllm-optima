# Loads the model with the given settings, then measures speed, memory and accuracy.
import os, sys, json, time, subprocess
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
from vllm import LLM, SamplingParams

QUESTIONS = [("What is 12 + 7? Number only.", "19"), ("Capital of France? One word.", "paris"),
             ("What is 9 times 3? Number only.", "27"), ("Opposite of hot? One word.", "cold")]
chat = lambda text: [{"role": "user", "content": text}]

if __name__ == "__main__":
    model, settings = sys.argv[1], json.loads(sys.argv[2])
    start = time.time()
    llm = LLM(model=model, **settings)

    try:
        context = llm.llm_engine.vllm_config.model_config.max_model_len
    except Exception:
        context = None

    def read(get):
        try:
            return get()
        except Exception:
            return None
    cfg = read(lambda: llm.llm_engine.vllm_config)
    engine = {"kv_cache_tokens": read(lambda: cfg.cache_config.num_gpu_blocks * cfg.cache_config.block_size),
              "max_num_seqs": read(lambda: cfg.scheduler_config.max_num_seqs),
              "max_num_batched_tokens": read(lambda: cfg.scheduler_config.max_num_batched_tokens),
              "gpu_memory_utilization": read(lambda: cfg.cache_config.gpu_memory_utilization)}

    load_s = time.time() - start
    memory = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True).stdout.split()[0]

    llm.chat([chat("Hello")] * 8, SamplingParams(max_tokens=16), use_tqdm=False)   # warm-up

    # Speed for one user: time per generated token (best of 3)
    one = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
    times = []
    for _ in range(3):
        start = time.time()
        llm.chat(chat("Explain the KV cache."), one, use_tqdm=False)
        times.append(time.time() - start)
    single = min(times)

    # Speed for many users: 64 requests at once
    start = time.time()
    outs = llm.chat([chat(f"A fact about the number {i}.") for i in range(64)], one, use_tqdm=False)
    tokens_per_s = sum(len(o.outputs[0].token_ids) for o in outs) / (time.time() - start)

    # Simple accuracy check: catches broken output, not a model benchmark
    answers = llm.chat([chat(q) for q, _ in QUESTIONS], SamplingParams(temperature=0, max_tokens=8),
                       use_tqdm=False)
    correct = sum(a in o.outputs[0].text.lower() for (_, a), o in zip(QUESTIONS, answers))

    print("RESULT:", json.dumps({"tokens_per_s": round(tokens_per_s), "ms_per_token": round(single / 128 * 1000, 1),
                                 "gpu_memory_gb": round(int(memory) / 1024, 1),
                                 "accuracy": f"{correct}/{len(QUESTIONS)}", "load_s": round(load_s),
                                 "context":context, "engine": engine}))