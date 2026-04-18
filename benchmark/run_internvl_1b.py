"""Benchmark runner for InternVL3-1B on single GPU (encoder_fission=False)."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import VisionArenaDataset
from schema import ExperimentConfig, VLLMConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit


APP_ID = "app-5d8f3c0e5b61405fbaefc6863ca17847"
MODEL_ID = "OpenGVLab/InternVL3-1B"


async def run() -> None:
    set_ulimit()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    dataset = VisionArenaDataset(
        dataset_path="lmarena-ai/VisionArena-Chat",
        dataset_split="train",
        random_seed=48105,
    )

    num_prompts = 200
    input_len = 256
    output_len = 128

    sampled = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_prompts,
        input_len=input_len,
        output_len=output_len,
    )
    print(f"Sampled {len(sampled)} requests from VisionArena.")

    backend_config = VLLMConfig(num_replicas=1, tp_size=1)

    for request_rate in [0.5, 1.0, 2.0]:
        config = ExperimentConfig(
            backend_config=backend_config,
            app_id=APP_ID,
            model_id=MODEL_ID,
            gpu_type="A40",
            num_gpus=1,
            num_prompts=num_prompts,
            num_warmups=5,
            input_len=input_len,
            output_len=output_len,
            request_rate=request_rate,
            burstiness=1.0,
            image_probability=1.0,
            image_width=640,
            image_height=480,
            image_count=1,
            image_choices=5,
            encoder_fission_probability=0.0,
            use_synthesized_data=True,
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