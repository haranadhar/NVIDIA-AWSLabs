# G6 L4 LLM serving PoC

This is a portfolio lab for an NVIDIA Cloud Solutions Engineer interview. It exposes an NVIDIA NIM (TensorRT-LLM-backed, OpenAI-compatible) LLM endpoint on an AWS G6 instance, then measures concurrent-user latency and throughput.

## What you can demonstrate

- Provisioning a production-shaped GPU host with IAM, VPC security boundaries, EBS storage, and a repeatable bootstrap.
- GPU and container diagnosis (`nvidia-smi`, CUDA container test, logs, health endpoint).
- Serving a model through NVIDIA software rather than a notebook-only demo.
- A customer decision based on p50/p95 latency, throughput, GPU memory/utilization, reliability, and cost per request.

## Architecture

```
load_test.py / client.py --> HTTPS reverse proxy (recommended) --> NIM container :8000
                                                               --> L4 GPU on g6.xlarge
NIM OpenAI API: /v1/chat/completions, /v1/models, /v1/health/ready
```

For a lab, port 8000 may be restricted to your own public IP. For a customer PoC, place the instance in a private subnet behind an ALB/API Gateway and authenticate callers; never expose an unauthenticated model endpoint to the internet.

## 1. Launch AWS resources

Use the AWS Console initially; it makes quotas, subnet choice, and cost visible. In a region where G6 capacity is available, launch:

- **Instance:** `g6.xlarge` (one NVIDIA L4, 24 GB VRAM, 4 vCPUs, 16 GiB RAM).
- **AMI:** current *Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 22.04*.
- **Storage:** 150 GB gp3 EBS (model/engine cache needs room).
- **Network:** public subnet only for a short lab; inbound SSH 22 and TCP 8000 **only from your public IP**. Production: SSM Session Manager + private subnet + ALB.
- **IAM role:** `AmazonSSMManagedInstanceCore`; add least-privilege S3 access only if you use S3 for logs/artifacts.

Before launch, request/verify the **Running On-Demand G and VT instances** quota in your selected region. Stop the instance whenever you are not measuring; terminate it after the lab if you do not need the disk.

SSH (replace the placeholders):

```bash
chmod 400 hara-gpu.pem
ssh -i hara-gpu.pem ubuntu@EC2_PUBLIC_DNS
```

## 2. Validate the GPU and Docker

```bash
nvidia-smi
docker --version
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
df -h
```

Expected outcome: `nvidia-smi` shows an NVIDIA L4. If Docker cannot see the GPU, do **not** move on: inspect `nvidia-smi`, the AMI driver, Docker service, and NVIDIA Container Toolkit first.

Optional live telemetry while generating traffic:

```bash
watch -n 1 'nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader'
```

## 3. Obtain credentials and choose an NIM

1. Create an NVIDIA NGC account and API key.
2. In the NGC catalog, select an LLM NIM that has an **L4-compatible profile** and confirm its current container tag, model license, and VRAM requirement.
3. Substitute that exact image for `NIM_IMAGE` below. NIM image names/tags and profiles change, so deliberately do not copy an old hard-coded tag from a blog.

```bash
export NGC_API_KEY='paste-your-ngc-api-key-here'
export NIM_IMAGE='nvcr.io/nim/<publisher>/<model>:<current-tag>'

echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
mkdir -p "$HOME/.cache/nim"
```

## 4. Run the model service

```bash
docker run --rm --name nim-llm --gpus all --shm-size=16GB \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -v "$HOME/.cache/nim:/opt/nim/.cache" \
  -p 8000:8000 \
  "$NIM_IMAGE"
```

First start can take time because it downloads/model-prepares artifacts. In another SSH terminal:

```bash
curl -s http://127.0.0.1:8000/v1/health/ready
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

Use the exact model ID returned by `/v1/models` when setting `MODEL` below.

## 5. Call it as an end user

Copy the two Python files to the EC2 instance and install only their small client dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export BASE_URL='http://127.0.0.1:8000'
export MODEL='MODEL_ID_FROM_V1_MODELS'
python client.py 'Explain why dynamic batching can improve GPU throughput in three bullets.'
```

The production caller must use a private URL/HTTPS endpoint and an authentication mechanism. `client.py` supports an optional bearer token through `API_KEY`; NIM local lab endpoints may not require one.

## 6. Benchmark concurrent users

Start conservatively so you understand the failure mode before increasing concurrency.

```bash
python load_test.py --requests 20 --concurrency 1 --max-tokens 100
python load_test.py --requests 60 --concurrency 4 --max-tokens 100
python load_test.py --requests 100 --concurrency 8 --max-tokens 100
```

Save each JSON result and pair it with a screenshot/log of `nvidia-smi`. Your comparison table should contain: instance, model/profile, prompt/output token settings, concurrency, p50/p95 latency, requests/sec, generated tokens/sec, GPU memory, GPU utilization, errors, and hourly instance price from your own AWS region.

## 7. G4dn vs G6 interview experiment

Repeat the exact workload on `g4dn.xlarge` only if the selected model/profile fits T4 VRAM. Keep the region, AMI generation, image tag, benchmark script, prompt, tokens, and concurrency identical.

Your recommendation should sound like this:

> The G4dn/T4 is the lower-cost baseline for smaller, lower-volume inference. The G6/L4 provided more memory headroom and better measured throughput at our target concurrency. I would select the G6 when the customer needs production concurrency or model growth; I would retain G4dn when its measured SLO and capacity are sufficient at lower cost.

Do not claim a universal price or performance ratio: availability and price vary by region, model, profile, prompt length, batching, and demand.

## 8. Troubleshooting drill (practice these aloud)

| Symptom | Check | Likely corrective action |
|---|---|---|
| `nvidia-smi` fails | AMI/driver, PCI device | Use supported DLAMI or install the supported driver; reboot if required. |
| Docker has no GPU | CUDA container test, Docker logs | Install/configure NVIDIA Container Toolkit; restart Docker. |
| OOM / service never ready | NIM logs, VRAM use, selected profile | Use an L4-compatible profile, lower context/concurrency, or select larger VRAM such as G6e. |
| Low GPU utilization, high p95 | CPU/network use, queueing, token settings | Profile preprocessing/network; increase useful concurrency or configured batching after validating latency SLO. |
| Errors under load | HTTP status/body, container logs | Reduce load, diagnose timeout/rate limits, and expose metrics before adding replicas. |

## Portfolio definition of done

Publish this repository with a redacted architecture diagram, reproducible launch checklist, benchmark CSV/JSON, one-page customer recommendation, and a 5-minute demo recording. Never publish API keys, public hostnames, customer data, or paid-model credentials.
