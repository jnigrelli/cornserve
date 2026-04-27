"""Task Dispatcher."""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
import uuid
from pathlib import Path
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import grpc
from opentelemetry import trace
from pydantic import BaseModel

from cornserve.logging import get_logger
from cornserve.services.pb import task_manager_pb2, task_manager_pb2_grpc
from cornserve.task.base import TASK_TIMEOUT, Stream, TaskInvocation, TaskOutput, UnitTask
from cornserve.task.forward import DataForward

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class OutputLengthPredictor:
    """Predicts output length bucket from a prompt embedding using the trained RLP model."""

    _MODEL_DIR = Path(__file__).parent / "predictor"

    def __init__(self) -> None:
        import joblib
        self._model = joblib.load(self._MODEL_DIR / "RLP-retrain.joblib")
        self._encoder = joblib.load(self._MODEL_DIR / "workload_encoder_retrain.joblib")
        logger.info("Loaded RLP model and encoder from %s", self._MODEL_DIR)

    def predict(self, invocations: list[TaskInvocation]) -> int:
        """Return predicted output length bucket (1–6, lower = shorter = higher SJF priority).

        Combines ML-predicted output length bucket with image count as a weighted penalty.
        Each extra image beyond the first adds 2 to the priority score.
        """
        import pandas as pd

        for inv in invocations:
            ti = inv.task_input
            embedding = ti.get("prompt_embedding") if isinstance(ti, dict) else getattr(ti, "prompt_embedding", None)
            image_count = ti.get("image_count", 1) if isinstance(ti, dict) else getattr(ti, "image_count", 1)

            ml_bucket = 1
            if embedding is not None:
                input_modality = ti.get("input_modality", "image") if isinstance(ti, dict) else getattr(ti, "input_modality", "image")
                output_modality = ti.get("output_modality", "text") if isinstance(ti, dict) else getattr(ti, "output_modality", "text")
                prompt_len = ti.get("prompt_len", 500) if isinstance(ti, dict) else getattr(ti, "prompt_len", 500)

                row = {
                    "input_modality": input_modality,
                    "output_modality": output_modality,
                    "input_len_bucket": prompt_len,
                    **{f"emb_{i}": v for i, v in enumerate(embedding)},
                }
                df = pd.DataFrame([row])
                encoded = self._encoder.transform(df)
                ml_bucket = int(self._model.predict(encoded)[0])

            image_penalty = (image_count - 1) * 2
            priority = ml_bucket + image_penalty
            logger.info("Predicted priority=%d (ml_bucket=%d, image_count=%d, penalty=%d)", priority, ml_bucket, image_count, image_penalty)
            return priority

        return 1  # fallback: degrades to FIFO


class TaskInfo:
    """Stores all task-related information.

    Attributes:
        task: The unit task object.
        task_manager_url: The URL to the task manager.
        task_manager_channel: The gRPC channel to the task manager.
        task_manager_stub: The gRPC stub to the task manager.
    """

    def __init__(self, task: UnitTask, task_manager_url: str) -> None:
        """Initialize the TaskInfo object."""
        self.task = task
        self.task_manager_url = task_manager_url
        self.task_manager_channel = grpc.aio.insecure_channel(task_manager_url)
        self.task_manager_stub = task_manager_pb2_grpc.TaskManagerStub(self.task_manager_channel)


@dataclass
class UnitTaskExecution:
    """Execution information for a single unit task invocation.

    Attributes:
        invocation: The task invocation object.
        executor_url: The URL to the task executor.
        executor_sidecar_ranks: The sidecar ranks for the task executor.
    """

    invocation: TaskInvocation
    executor_url: str
    executor_sidecar_ranks: list[int]

    @property
    def is_streaming(self) -> bool:
        """Check if the task streams its output."""
        return isinstance(self.invocation.task_output, Stream)


def iter_data_forwards(obj: object) -> Iterator[DataForward]:
    """Recursively find and iterate through all DataForward objects.

    This method knows how to flatten list, tuple, dict, and nested BaseModel objects.

    Args:
        obj: The object to search for DataForward objects.

    Yields:
        All DataForward objects found in the model's field values
    """
    if isinstance(obj, DataForward):
        yield obj
    elif isinstance(obj, list | tuple):
        for item in obj:
            yield from iter_data_forwards(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from iter_data_forwards(item)
    elif isinstance(obj, BaseModel):
        # Recursively search through nested BaseModels. Make sure references to the original
        # `DataForward` objects are yielded, so that external mutations are reflected.
        for name in obj.__class__.model_fields:
            yield from iter_data_forwards(getattr(obj, name))


AGING_INTERVAL_SECS: float = 60.0   # reduce effective priority by 1 every N seconds of waiting
AGING_MIN_PRIORITY: int = 1         # floor — no request can be boosted past highest priority

@dataclass(order=True)
class _SJFEntry:
    """Priority queue entry. Ordered by (effective_priority, arrival_seq).

    effective_priority starts at original_priority and decreases over time (aging),
    preventing starvation of heavy requests.
    """
    effective_priority: int
    arrival_seq: int
    original_priority: int = field(compare=False)
    enqueue_time: float = field(compare=False)
    invocations: list[TaskInvocation] = field(compare=False)
    future: asyncio.Future = field(compare=False)

    def apply_aging(self, now: float) -> None:
        """Decrease effective_priority based on time waited."""
        age_secs = now - self.enqueue_time
        decrement = int(age_secs / AGING_INTERVAL_SECS)
        self.effective_priority = max(AGING_MIN_PRIORITY, self.original_priority - decrement)


class TaskDispatcher:
    """Task Dispatcher."""

    def __init__(self) -> None:
        """Initialize the Task Dispatcher."""
        self.task_lock = asyncio.Lock()
        self.task_infos: dict[str, TaskInfo] = {}

        self.ongoing_task_lock = asyncio.Lock()
        self.ongoing_invokes: dict[str, list[asyncio.Task]] = defaultdict(list)
        self.client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=TASK_TIMEOUT),
            connector=aiohttp.TCPConnector(limit=0),
        )

        self.predictor = OutputLengthPredictor()
        self._sjf_heap: list[_SJFEntry] = []
        self._heap_event: asyncio.Event = asyncio.Event()
        self._arrival_counter = itertools.count()
        self._worker: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the SJF worker. Must be called from an async context after the event loop is running."""
        self._worker = asyncio.create_task(self._sjf_worker())

    async def notify_task_deployment(self, task: UnitTask, task_manager_url: str) -> None:
        """Register a newly deployed task and its task manager with the dispatcher."""
        async with self.task_lock:
            self.task_infos[task.id] = TaskInfo(task, task_manager_url)

        logger.info(
            "Registered new task %s(%s) with task manager URL %s",
            task.__class__.__name__,
            task,
            task_manager_url,
        )

    async def notify_task_teardown(self, task: UnitTask) -> None:
        """Remove a task that has been torn down.

        This will cancel all ongoing invokes for the task.
        """
        async with self.task_lock:
            if task.id not in self.task_infos:
                raise ValueError(f"Task {task} not found in task dispatcher.")

            task_info = self.task_infos.pop(task.id)

        # Cancel all ongoing invokes for the task
        async with self.ongoing_task_lock:
            for invoke_task in self.ongoing_invokes.pop(task.id, []):
                invoke_task.cancel()

        # Close the gRPC channel to the task manager
        await task_info.task_manager_channel.close()

        logger.info("Removed task %s from task dispatcher", task)

    async def shutdown(self) -> None:
        """Shutdown the Task Dispatcher."""
        coros = []
        for task_info in self.task_infos.values():
            coros.append(task_info.task_manager_channel.close())

        results = await asyncio.gather(*coros, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.exception("Error occured while shutting down task dispatcher: %s", result)

        if self._worker is not None:
            self._worker.cancel()
            self._worker = None

        await self.client.close()

        logger.info("Task dispatcher shutdown complete")

    async def _sjf_worker(self) -> None:
        """Process requests in SJF order with aging to prevent starvation."""
        while True:
            await self._heap_event.wait()
            if not self._sjf_heap:
                self._heap_event.clear()
                continue

            # Apply aging to all waiting entries, then re-heapify
            now = time.monotonic()
            for entry in self._sjf_heap:
                entry.apply_aging(now)
            heapq.heapify(self._sjf_heap)

            entry = heapq.heappop(self._sjf_heap)
            if not self._sjf_heap:
                self._heap_event.clear()

            wait_secs = now - entry.enqueue_time
            logger.info(
                "Dispatching entry original_priority=%d effective_priority=%d wait=%.1fs queue_remaining=%d",
                entry.original_priority, entry.effective_priority, wait_secs, len(self._sjf_heap),
            )
            try:
                result = await self._dispatch(entry.invocations)
                entry.future.set_result(result)
            except Exception as e:
                entry.future.set_exception(e)

    @tracer.start_as_current_span("TaskDispatcher.invoke")
    async def invoke(self, invocations: list[TaskInvocation]) -> list[TaskOutput]:
        """Enqueue invocations under SJF with aging and block until the worker executes them."""
        predicted_len = self.predictor.predict(invocations)
        future: asyncio.Future[list[TaskOutput]] = asyncio.get_event_loop().create_future()
        entry = _SJFEntry(
            effective_priority=predicted_len,
            arrival_seq=next(self._arrival_counter),
            original_priority=predicted_len,
            enqueue_time=time.monotonic(),
            invocations=invocations,
            future=future,
        )
        heapq.heappush(self._sjf_heap, entry)
        self._heap_event.set()
        logger.info("Enqueued request with priority=%d (queue size=%d)", predicted_len, len(self._sjf_heap))
        return await future

    async def _dispatch(self, invocations: list[TaskInvocation]) -> list[TaskOutput]:
        """Execute a full request graph: route → wire DataForwards → run concurrently."""
        span = trace.get_current_span()
        span.set_attributes(
            {
                f"task_dispatcher.invoke.invocations.{i}": invocation.model_dump_json()
                for i, invocation in enumerate(invocations)
            }
        )

        # Check if all tasks are registered with the dispatcher
        task_infos: list[TaskInfo] = []
        async with self.task_lock:
            logger.info("TaskDispatcher: registered tasks: %s", list(self.task_infos.keys()))
            for invocation in invocations:
                for task_id, task_info in self.task_infos.items():
                    eq = task_info.task.is_equivalent_to(invocation.task)
                    logger.info(
                        "TaskDispatcher: compare invocation task=%r to registered task_id=%s eq=%s",
                        invocation.task,
                        task_id,
                        eq,
                    )
                    if eq:
                        task_infos.append(task_info)
                        break
                else:
                    logger.error("Task not found for invocation %s (no equivalent registered task)", invocation)
                    raise ValueError(f"Task {invocation.task} not found in task dispatcher.")
        assert len(task_infos) == len(invocations), "Task info count mismatch"

        # Get task executor routes and sidecar ranks
        task_executions: list[UnitTaskExecution] = []
        get_route_coros: list[asyncio.Future[task_manager_pb2.GetRouteResponse]] = []
        request_id = uuid.uuid4().hex
        for task_info in task_infos:
            get_route_coros.append(
                task_info.task_manager_stub.GetRoute(task_manager_pb2.GetRouteRequest(request_id=request_id))
            )
        for invocation, route_response in zip(invocations, await asyncio.gather(*get_route_coros), strict=True):
            task_executions.append(
                UnitTaskExecution(
                    invocation=invocation,
                    executor_url=route_response.task_executor_url,
                    executor_sidecar_ranks=list(route_response.sidecar_ranks),
                )
            )

        # TODO: Build an actual graph, submit to a centralized asyncio.Task that does scheduling.

        # Dig up `DataForward` objects and connect producers and consumers.
        #
        # Example: Two encoders embed two images, and both are sent to two separate LLMs.
        #
        #                   task_input           task_output
        #    Encoder1                         DataForward(id=1)
        #    Encoder2                         DataForward(id=2)
        #        LLM1  DataForward(id=1,2)
        #        LLM2  DataForward(id=1,2)
        #
        # We iterate through `DataForward` objects in the order of invocations, and within each invocation,
        # those in the input and then those in the output. Ultimately, the source and destination sidecar
        # ranks of connected (i.e., same ID) `DataForward` objects should be identical.
        # When we see a `DataForward` object in the output, it is a producer, and from the invocation's
        # task executor routing result, we can figure out its source sidecar ranks.
        # When we see a `DataForward` object in the input, it is a consumer, and it *must* have been
        # previously encountered in the output of a previous invocation. From the invocation's task executor
        # routing result, we can figure out its destination sidecar ranks.
        # Note that when we inplace set the sidecar ranks in `DataForward` objects, we are doing so on
        # references to the original `DataForward` objects in the task input and output.
        producer_forwards: dict[str, DataForward] = {}
        for execution in task_executions:
            # Iterate recursively over all `DataForward` objects in the task input.
            # Encountered `DataForward` objects are consumers, which should have been encountered
            # previously in earlier task invocations. If not, it's an error.
            for consumer_forward in iter_data_forwards(execution.invocation.task_input):
                try:
                    producer_forward = producer_forwards[consumer_forward.id]
                except KeyError as e:
                    raise ValueError(
                        f"Consumer `DataForward[{consumer_forward.data_type}](id={consumer_forward.id})` in the "
                        f"input of invocation {execution.invocation} not found in previous task invocations."
                    ) from e
                assert producer_forward.src_sidecar_ranks is not None
                consumer_forward.src_sidecar_ranks = producer_forward.src_sidecar_ranks
                if producer_forward.dst_sidecar_ranks is None:
                    producer_forward.dst_sidecar_ranks = []
                producer_forward.dst_sidecar_ranks.append(execution.executor_sidecar_ranks)
                consumer_forward.dst_sidecar_ranks = producer_forward.dst_sidecar_ranks

            # Iterate recursively over all `DataForward` objects in the task output.
            # Encountered `DataForward` objects are producers, which we save in `data_forwards`.
            for producer_forward in iter_data_forwards(execution.invocation.task_output):
                producer_forward.src_sidecar_ranks = execution.executor_sidecar_ranks
                producer_forwards[producer_forward.id] = producer_forward

        logger.info("Connected all DataForward objects in task invocations")

        # Verify whether all `DataForward` objects are properly connected
        for data_forward in producer_forwards.values():
            assert data_forward.src_sidecar_ranks is not None
            assert data_forward.dst_sidecar_ranks is not None

        # Dispatch all task invocations to task executors
        dispatch_coros: list[asyncio.Task[TaskOutput]] = []
        try:
            async with asyncio.TaskGroup() as tg:
                for execution in task_executions:
                    # `TaskInput` -> JSON request to task executor
                    request = execution.invocation.task.execution_descriptor.to_request(
                        task_input=execution.invocation.task_input,
                        task_output=execution.invocation.task_output,
                    )
                    dispatch_coros.append(tg.create_task(self._execute_unit_task(execution, request)))
        except (ExceptionGroup, Exception) as e:
            logger.exception("Error while invoking task")
            if isinstance(e, ExceptionGroup):
                raise RuntimeError(f"Task invocation failed: {e.exceptions}") from e
            else:
                raise RuntimeError(f"Task invocation failed: {e}") from e

        # Collect responses from task executors
        return [task.result() for task in dispatch_coros]

    async def _execute_unit_task(self, execution: UnitTaskExecution, request: dict[str, Any]) -> TaskOutput:
        """Execute a single task by sending request to executor and processing response."""
        url = execution.invocation.task.execution_descriptor.get_api_url(execution.executor_url)
        logger.debug(
            "Invoking %s task %s by posting request %s to %s",
            "streaming" if execution.is_streaming else "non-streaming",
            execution.invocation.task.__class__.__name__,
            request,
            url,
        )
        try:
            response = await self.client.post(url, json=request)
            response.raise_for_status()
            logger.debug(
                "Task %s response: %s",
                execution.invocation.task.__class__.__name__,
                "[Stream]" if execution.is_streaming else await response.text(),
            )
        except aiohttp.ClientResponseError as e:
            logger.exception("Error while invoking task")
            raise RuntimeError(
                f"HTTP request failed with code {e.status}: {e.message}",
            ) from e
        except Exception as e:
            logger.exception("Error while invoking task")
            raise RuntimeError(f"HTTP request failed: {e}") from e

        # HTTP response from the Task Executor is converted to TaskOutput.
        # For streaming tasks, `task_output` is a Stream object.
        task_output: TaskOutput = await execution.invocation.task.execution_descriptor.from_response(
            task_output=execution.invocation.task_output,
            response=response,
        )
        return task_output
