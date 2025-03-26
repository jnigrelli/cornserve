# Sidecar developer guide

## Docker container

It is recommended to run everything inside docker. That said, sidecar has minimal
dependencies and _could_ run directly on the host.

Specifying `--shm-size` with at least 2 GB and `--ipc host` is required.

## Editable installation

```bash
pip install -e 'python[dev]'
```

## Testing

We use pytest.

```bash
pytest python/tests/services/sidecar/test_sidecar.py
```
