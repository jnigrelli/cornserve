"""Output-length HOL blocking benchmark.

Splits requests by predicted output bucket (low=short, high=long), assigns
matching max_completion_tokens, then sends long requests first to induce HOL.
Under FIFO, short requests wait behind long ones. Under SJF, short ones are
served first. This gives a clean comparison where predictor scores match reality.
"""

import asyncio
import os
import pickle
from dataclasses import dataclass, field
from typing import Any
import sys

sys.path.insert(0, os.path.dirname(__file__))

from benchmark_cornserve import RequestInput, cornserve_invoke
from schema import ExperimentConfig, VLLMConfig
from tqdm import tqdm

from cornserve.utils import set_ulimit

APP_ID = "app-3823822d35514d68a19ab549b0043afc"
MODEL_ID = "OpenGVLab/InternVL3-1B"
WORKLOAD_PATH = "/home/jnigrelli/cse585/frozen_vision_workload.pkl"
PREDICTOR_DIR = "/home/jnigrelli/cse585/cornserve/python/cornserve/services/task_dispatcher/predictor"

SHORT_TOKENS = 50    # max_completion_tokens for predicted-short requests
LONG_TOKENS = 300    # max_completion_tokens for predicted-long requests
PAUSE_BETWEEN_BURSTS = 1.0  # seconds; long requests enter queue before short ones arrive


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


class OutputHOLVLLMConfig(VLLMConfig):
    def to_subdir_name(self) -> str:
        return f"vllm+replicas{self.num_replicas}+tp{self.tp_size}+output_hol"


def load_workload(path: str, num_prompts: int):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data[:num_prompts]


def predict_bucket(request: SampleRequest) -> int:
    """Run local predictor to get output length bucket for a request."""
    import joblib
    import pandas as pd

    model = joblib.load(f"{PREDICTOR_DIR}/RLP-retrain.joblib")
    encoder = joblib.load(f"{PREDICTOR_DIR}/workload_encoder_retrain.joblib")

    emb = request.prompt_embedding
    if emb is None:
        return 3
    if isinstance(emb, pd.DataFrame):
        embedding = emb.values[0].tolist()
    elif hasattr(emb, "tolist"):
        embedding = emb.tolist()
    else:
        embedding = list(emb)

    row = {
        "input_modality": request.input_modality,
        "output_modality": request.output_modality,
        "input_len_bucket": request.prompt_len,
        **{f"emb_{i}": v for i, v in enumerate(embedding)},
    }
    df = pd.DataFrame([row])
    encoded = encoder.transform(df)
    return int(model.predict(encoded)[0])


def make_request_input(request: SampleRequest, config: ExperimentConfig, output_len: int) -> RequestInput:
    import pandas as pd

    emb = request.prompt_embedding
    embedding = None
    if emb is not None:
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
        output_len=output_len,
        multi_modal_data=[request.multi_modal_data],
        filenames=request.filenames,
        encoder_fission=False,
        prompt_embedding=embedding,
    )


async def run_output_hol_benchmark(config: ExperimentConfig, sampled: list) -> None:
    """Predict buckets, split into short/long, send long first then short."""
    import joblib

    # Load models once
    print("Running predictor on workload...")
    buckets = [predict_bucket(r) for r in sampled]
    median_bucket = sorted(buckets)[len(buckets) // 2]
    print(f"  Bucket distribution: {sorted(set(buckets))}, median={median_bucket}")

    short_samples = [(r, b) for r, b in zip(sampled, buckets) if b <= median_bucket]
    long_samples = [(r, b) for r, b in zip(sampled, buckets) if b > median_bucket]

    # If predictor returns same bucket for everything, split by index
    if not long_samples or not short_samples:
        print("  Predictor returned uniform buckets — splitting by index instead")
        mid = len(sampled) // 2
        long_samples = [(r, 3) for r in sampled[:mid]]
        short_samples = [(r, 3) for r in sampled[mid:]]

    print(f"  Long requests: {len(long_samples)} → {LONG_TOKENS} tokens")
    print(f"  Short requests: {len(short_samples)} → {SHORT_TOKENS} tokens")

    long_inputs = [make_request_input(r, config, LONG_TOKENS) for r, _ in long_samples]
    short_inputs = [make_request_input(r, config, SHORT_TOKENS) for r, _ in short_samples]

    # Test first
    print("Testing a single request...")
    test_out = await cornserve_invoke(short_inputs[0], tqdm(total=1, desc="Test"))
    if not test_out.success:
        raise RuntimeError(f"Test request failed: {test_out.error}")
    print("Test passed.")

    pbar = tqdm(total=len(long_inputs) + len(short_inputs), desc="Output HOL benchmark")

    # Fire long burst first
    long_tasks = [asyncio.create_task(cornserve_invoke(inp, pbar)) for inp in long_inputs]

    # Pause so long requests are queued before short ones arrive
    await asyncio.sleep(PAUSE_BETWEEN_BURSTS)

    # Fire short requests
    short_tasks = [asyncio.create_task(cornserve_invoke(inp, pbar)) for inp in short_inputs]

    long_results = await asyncio.gather(*long_tasks)
    short_results = await asyncio.gather(*short_tasks)
    pbar.close()

    all_results = list(long_results) + list(short_results)
    successes = sum(1 for r in all_results if r.success)
    failures = [r for r in all_results if not r.success]
    print(f"\nCompleted: {successes} success, {len(failures)} failure")
    if failures:
        for f in failures[:3]:
            print(f"  Error: {f.error}")

    long_latencies = sorted(r.latency for r in long_results if r.success and r.latency)
    short_latencies = sorted(r.latency for r in short_results if r.success and r.latency)

    def p(lat, pct):
        return lat[int(len(lat) * pct)] if lat else float("nan")

    if long_latencies:
        print(f"Long  P50={p(long_latencies, 0.50):.2f}s  P95={p(long_latencies, 0.95):.2f}s  P99={p(long_latencies, 0.99):.2f}s")
    if short_latencies:
        print(f"Short P50={p(short_latencies, 0.50):.2f}s  P95={p(short_latencies, 0.95):.2f}s  P99={p(short_latencies, 0.99):.2f}s")


async def run() -> None:
    set_ulimit()

    num_prompts = 100
    sampled = load_workload(WORKLOAD_PATH, num_prompts)
    print(f"Loaded {len(sampled)} requests.")

    backend_config = OutputHOLVLLMConfig(num_replicas=1, tp_size=1)
    config = ExperimentConfig(
        backend_config=backend_config,
        app_id=APP_ID,
        model_id=MODEL_ID,
        gpu_type="A40",
        num_gpus=1,
        num_prompts=num_prompts,
        num_warmups=0,
        input_len=sampled[0].prompt_len,
        output_len=LONG_TOKENS,
        request_rate=1.0,
        burstiness=1.0,
        image_probability=1.0,
        image_count=1,
        image_choices=1,
        encoder_fission_probability=0.0,
        use_synthesized_data=False,
    )

    await run_output_hol_benchmark(config, sampled)


if __name__ == "__main__":
    asyncio.run(run())