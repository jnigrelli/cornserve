"""HOL blocking benchmark: mix of 3-image (heavy) and 1-image (light) requests.

Heavy requests arrive first in a burst, then light requests follow after a short
pause. This induces head-of-line blocking under FIFO and shows SJF benefit.
Starvation is avoided by capping the heavy burst size and using a modest pause.
"""

import asyncio
import os
import pickle
import copy
from dataclasses import dataclass, field
from typing import Any
import sys

sys.path.insert(0, os.path.dirname(__file__))

from benchmark_cornserve import RequestInput, cornserve_invoke
from schema import ExperimentConfig, VLLMConfig
from tqdm import tqdm

from cornserve.utils import set_ulimit

APP_ID = "app-3c2dfcc93e864b0289f629a4c8d38b11"
MODEL_ID = "OpenGVLab/InternVL3-1B"
WORKLOAD_PATH = "/home/jnigrelli/cse585/frozen_vision_workload.pkl"

HEAVY_IMAGE_COPIES = 2  # duplicates per heavy request; stay within 8192 token limit
HEAVY_FRACTION = 0.5    # 30% heavy, 70% light
PAUSE_BETWEEN_BURSTS = 1  # seconds between heavy burst and light wave (avoids starvation)


@dataclass
class SampleRequest:
    prompt: str | Any
    prompt_len: int
    expected_output_len: int
    multi_modal_data: dict[str, Any]
    image_urls: list[str] = field(default_factory=list)
    filenames: list[str] = field(default_factory=list)
    encoder_fission: bool = False
    prompt_embedding: Any = False
    input_modality: str = "image"
    output_modality: str = "text"


class HOLVLLMConfig(VLLMConfig):
    def to_subdir_name(self) -> str:
        return f"vllm+replicas{self.num_replicas}+tp{self.tp_size}+hol"


def load_workload(path: str, num_prompts: int):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data[:num_prompts]


def make_request_input(request: SampleRequest, config: ExperimentConfig, num_copies: int = 1) -> RequestInput:
    """Build a RequestInput, duplicating the image num_copies times."""
    import pandas as pd

    base_mm = request.multi_modal_data
    mm_data_list = [base_mm] * num_copies

    embedding = None
    if hasattr(request, "prompt_embedding") and request.prompt_embedding is not None:
        emb = request.prompt_embedding
        if isinstance(emb, pd.DataFrame):
            embedding = emb.values[0].tolist()
        elif hasattr(emb, "tolist"):
            embedding = emb.tolist()
        else:
            embedding = list(emb)

    return RequestInput(
        url=f"http://127.0.0.1:30080/app/invoke/{config.app_id}",
        model=config.model_id,
        prompt=request.prompt,
        prompt_len=request.prompt_len,
        output_len=config.output_len,
        multi_modal_data=mm_data_list,
        filenames=request.filenames,
        encoder_fission=False,
        prompt_embedding=embedding,
        image_count=num_copies,
    )


async def run_hol_benchmark(config: ExperimentConfig, sampled: list) -> None:
    """Send heavy requests first, pause, then send light requests concurrently."""
    num_heavy = int(len(sampled) * HEAVY_FRACTION)
    heavy_samples = sampled[:num_heavy]
    light_samples = sampled[num_heavy:]

    heavy_inputs = [make_request_input(r, config, HEAVY_IMAGE_COPIES) for r in heavy_samples]
    light_inputs = [make_request_input(r, config, 1) for r in light_samples]

    print(f"  Heavy requests: {len(heavy_inputs)} (x{HEAVY_IMAGE_COPIES} images each)")
    print(f"  Light requests: {len(light_inputs)} (x1 image each)")
    print(f"  Pause between bursts: {PAUSE_BETWEEN_BURSTS}s")

    results = []
    pbar = tqdm(total=len(heavy_inputs) + len(light_inputs), desc="HOL benchmark")

    # Test single request first
    print("Testing a single request...")
    test_out = await cornserve_invoke(light_inputs[0], tqdm(total=1, desc="Test"))
    if not test_out.success:
        raise RuntimeError(f"Test request failed: {test_out.error}")
    print("Test passed.")

    # Fire heavy burst
    heavy_tasks = [asyncio.create_task(cornserve_invoke(inp, pbar)) for inp in heavy_inputs]

    # Brief pause so heavy requests enter the queue before light ones
    await asyncio.sleep(PAUSE_BETWEEN_BURSTS)

    # Fire light requests
    light_tasks = [asyncio.create_task(cornserve_invoke(inp, pbar)) for inp in light_inputs]

    heavy_results = await asyncio.gather(*heavy_tasks)
    light_results = await asyncio.gather(*light_tasks)
    pbar.close()

    results = list(heavy_results) + list(light_results)
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    print(f"\nCompleted: {len(successes)} success, {len(failures)} failure")
    if failures:
        for f in failures[:3]:
            print(f"  Error: {f.error}")

    def stats(label, results):
        ok = [r for r in results if r.success]
        if not ok:
            print(f"{label}: no successful results")
            return
        lat = sorted(r.latency for r in ok)
        ttft = sorted(r.ttft for r in ok if r.ttft)
        tpot = sorted(r.tpot for r in ok if r.tpot)
        tokens = [r.output_tokens for r in ok if r.output_tokens]

        def p(xs, pct): return xs[int(len(xs) * pct)] if xs else float("nan")

        duration = max(r.completion_timestamp for r in ok) - min(r.start_timestamp for r in ok)
        req_throughput = len(ok) / duration if duration > 0 else 0
        tok_throughput = sum(tokens) / duration if duration > 0 and tokens else 0

        print(f"\n{label} ({len(ok)} reqs):")
        print(f"  E2E latency  — P50={p(lat,0.50):.2f}s  P95={p(lat,0.95):.2f}s  P99={p(lat,0.99):.2f}s  mean={sum(lat)/len(lat):.2f}s")
        if ttft:
            print(f"  TTFT         — P50={p(ttft,0.50):.3f}s  P95={p(ttft,0.95):.3f}s  P99={p(ttft,0.99):.3f}s  mean={sum(ttft)/len(ttft):.3f}s")
        if tpot:
            print(f"  TPOT         — P50={p(tpot,0.50)*1000:.1f}ms  P95={p(tpot,0.95)*1000:.1f}ms  mean={sum(tpot)/len(tpot)*1000:.1f}ms")
        if tokens:
            print(f"  Output tokens — mean={sum(tokens)/len(tokens):.1f}  min={min(tokens)}  max={max(tokens)}")
        print(f"  Throughput   — {req_throughput:.2f} req/s  {tok_throughput:.1f} tok/s")

    stats("Heavy", heavy_results)
    stats("Light", light_results)


async def run() -> None:
    set_ulimit()

    num_prompts = 100
    sampled = load_workload(WORKLOAD_PATH, num_prompts)
    print(f"Loaded {len(sampled)} requests.")

    backend_config = HOLVLLMConfig(num_replicas=1, tp_size=1)
    config = ExperimentConfig(
        backend_config=backend_config,
        app_id=APP_ID,
        model_id=MODEL_ID,
        gpu_type="A40",
        num_gpus=1,
        num_prompts=num_prompts,
        num_warmups=0,
        input_len=sampled[0].prompt_len,
        output_len=sampled[0].expected_output_len,
        request_rate=1.0,
        burstiness=1.0,
        image_probability=1.0,
        image_count=1,
        image_choices=1,
        encoder_fission_probability=0.0,
        use_synthesized_data=False,
    )

    await run_hol_benchmark(config, sampled)


if __name__ == "__main__":
    asyncio.run(run())