# SJF Scheduling for Cornserve — Benchmark Guide

This fork adds **Shortest-Job-First (SJF) scheduling with priority aging** to the Cornserve task dispatcher. The benchmark demonstrates head-of-line (HOL) blocking reduction on mixed image workloads.

---

## Prerequisites

- A running Kubernetes cluster (k3s or Minikube) with at least one GPU node
- Cornserve deployed — follow the [official getting started guide](https://cornserve.ai/getting_started/) to get the stack up
- Python 3.11+ with `uv` available
- A frozen workload pickle at the path set in `WORKLOAD_PATH` (see below)

---

## 1. Deploy Cornserve

Follow [cornserve.ai/getting_started](https://cornserve.ai/getting_started/) to deploy the core services. The stack consists of:

| Service | Role |
|---|---|
| `gateway` | HTTP entry point, routes app invocations |
| `task-dispatcher` | Schedules tasks across executors (SJF logic lives here) |
| `resource-manager` | Tracks GPU resources and task executor pods |
| `sidecar` | Node-level agent, manages executor lifecycle |

After deploying, verify all pods are running:

```bash
kubectl get pods -n cornserve
```

### Restart order (important)

If you restart `resource-manager`, always kill the sidecar first or the resource-manager will fail to re-register:

```bash
kubectl delete pod sidecar-0 -n cornserve
kubectl wait --for=condition=Ready pod/sidecar-0 -n cornserve --timeout=60s
kubectl rollout restart deployment/resource-manager -n cornserve
```

---

## 2. Register the App

Register the InternVL3-1B app and note the returned `APP_ID`:

```bash
cd benchmark
uv run app_utils.py
```

Copy the printed `APP_ID` into `run_hol_benchmark.py`:

```python
APP_ID = "app-<your-id-here>"
```

---

## 3. Prepare the Workload

The benchmark uses a frozen pickle of pre-processed VisionArena requests (prompts + image embeddings). Generate it once:

```bash
cd benchmark
uv run benchmark_dataset.py --output /path/to/frozen_vision_workload.pkl
```

Then set `WORKLOAD_PATH` in `run_hol_benchmark.py` to that path.

---

## 4. Run the HOL Blocking Benchmark

The benchmark sends 50 heavy (2-image) requests in a burst, waits 1 second, then sends 50 light (1-image) requests. This induces HOL blocking under FIFO and shows the SJF benefit.

```bash
cd benchmark
uv run run_hol_benchmark.py
```

Output includes per-group TTFT, E2E latency, and throughput:

```
Heavy requests: 50 (x2 images each)
Light requests: 50 (x1 image each)

Heavy (50 reqs):
  TTFT  — P50=40.16s  P95=68.14s  mean=41.69s
  ...
Light (50 reqs):
  TTFT  — P50=13.88s  P95=28.62s  mean=14.19s
  ...
```

### Switching between FIFO and SJF

The scheduling policy is controlled in the task dispatcher. To run a FIFO baseline, set the predictor to return a constant in `dispatcher.py`:

```python
# In OutputLengthPredictor.predict(), return a fixed value to simulate FIFO:
return 1
```

Patch and redeploy the dispatcher container after any change:

```bash
sudo nerdctl --namespace k8s.io run --net=none -d --name tmp-td \
  --entrypoint /bin/sleep cornserve/task-dispatcher:latest 360000

sudo nerdctl --namespace k8s.io cp \
  python/cornserve/services/task_dispatcher/dispatcher.py \
  tmp-td:/workspace/cornserve/python/cornserve/services/task_dispatcher/dispatcher.py

sudo nerdctl --namespace k8s.io commit tmp-td cornserve/task-dispatcher:latest
sudo nerdctl --namespace k8s.io rm -f tmp-td

kubectl rollout restart deployment/task-dispatcher -n cornserve
kubectl rollout status deployment/task-dispatcher -n cornserve
```

---

## 5. Tuning

Key constants in `dispatcher.py`:

| Constant | Default | Effect |
|---|---|---|
| `AGING_INTERVAL_SECS` | `30.0` | Seconds between priority decrements; longer = less throughput impact |
| `AGING_MIN_PRIORITY` | `1` | Floor on effective priority; prevents complete demotion |

Key constants in `run_hol_benchmark.py`:

| Constant | Default | Effect |
|---|---|---|
| `HEAVY_IMAGE_COPIES` | `2` | Images per heavy request |
| `HEAVY_FRACTION` | `0.5` | Fraction of requests that are heavy |
| `PAUSE_BETWEEN_BURSTS` | `1` | Seconds between heavy burst and light wave |