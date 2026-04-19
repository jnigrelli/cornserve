"""Benchmark runner for InternVL3-1B on single GPU (encoder_fission=False)."""

import asyncio
import os
import pickle
from dataclasses import dataclass, field
from typing import Any
import sys

sys.path.insert(0, os.path.dirname(__file__))

from benchmark_cornserve import benchmark, transform_sampled_requests
from schema import ExperimentConfig, VLLMConfig


class SJFVLLMConfig(VLLMConfig):
    def to_subdir_name(self) -> str:
        return f"vllm+replicas{self.num_replicas}+tp{self.tp_size}+sjf"

from cornserve.utils import set_ulimit

APP_ID = "app-f15855b270b743c4ad552785ce4b7362"
MODEL_ID = "OpenGVLab/InternVL3-1B"
WORKLOAD_PATH = "/home/jnigrelli/cse585/frozen_vision_workload.pkl"


@dataclass
class SampleRequest:
    """Represents a single inference request for benchmarking."""

    prompt: str | Any
    prompt_len: int
    expected_output_len: int
    # original image
    multi_modal_data: dict[str, Any]
    # synthetic data
    image_urls: list[str] = field(default_factory=list)
    filenames: list[str] = field(default_factory=list)
    encoder_fission: bool = False
    prompt_embedding: Any = False
    input_modality: str = "image"
    output_modality: str = "text"

def load_workload(path: str, num_prompts: int):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data[:num_prompts]


async def run() -> None:
    set_ulimit()

    num_prompts = 200

    sampled = load_workload(WORKLOAD_PATH, num_prompts)
    print(f"Loaded {len(sampled)} requests from frozen workload.")
    print(f"  prompt_len={sampled[0].prompt_len}, expected_output_len={sampled[0].expected_output_len}")
    print(f"  prompt_embedding shape: {sampled[0].prompt_embedding.shape}")

    backend_config = SJFVLLMConfig(num_replicas=1, tp_size=1)

    for request_rate in [0.5, 1.0, 2.0]:
        config = ExperimentConfig(
            backend_config=backend_config,
            app_id=APP_ID,
            model_id=MODEL_ID,
            gpu_type="A40",
            num_gpus=1,
            num_prompts=num_prompts,
            num_warmups=5,
            input_len=sampled[0].prompt_len,
            output_len=sampled[0].expected_output_len,
            request_rate=request_rate,
            burstiness=1.0,
            image_probability=1.0,
            image_count=1,
            image_choices=1,
            encoder_fission_probability=0.0,
            use_synthesized_data=False,  # use the real base64 images from the pickle
        )

        if config.exists():
            print(f"Skipping rate={request_rate} — results already exist at {config.to_path()}")
            continue

        print(f"\n{'='*50}")
        print(f"Running benchmark at request_rate={request_rate} req/s")
        request_inputs = transform_sampled_requests(config, sampled)
        await benchmark(request_inputs, config)


if __name__ == "__main__":
    asyncio.run(run())