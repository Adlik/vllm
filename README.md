
## Requirements

vLLM is a Python library that also contains pre-compiled C++ and CUDA (12.1) binaries.

OS: Linux

Python: 3.8 – 3.11

GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

## Deploying with Docker
### Build from source
```bash
DOCKER_BUILDKIT=1 docker build -t vllm_adlik_build:latest .
```
### Create serving container
```bash
docker run -it --gpus all \
    -v /path/to/model:/model \
    -p 48000:48000 \
    vllm_adlik_build:latest \
    bash
```
### Serving deployment
```bash
# Export env params
export VLLM_SERVING_IDX="vllm-serving"

# Service (int4 quantization)
python3 -m vllm.entrypoints.api_server --model /model --tensor-parallel-size 2 --host 0.0.0.0 --port 48000 --trust-remote-code --max-num-batched-tokens 16384 --max-model-len 16384 --dtype half --quantization autoquant --max-num-seqs 16 --enforce-eager
```
After successful deployment of the service, it can be accessed through the following interfaces:
```
"<server-ip>:48000/generate"
```
