# AWS practice roadmap for NVIDIA Cloud / AI Infrastructure roles

Every lab must result in a proof artifact: architecture, command output, metric, diagnosis, recommendation, or customer presentation.

| NVIDIA requirement | AWS proof |
|---|---|
| Customer PoC and inference software | G6 L4 NIM/TensorRT-LLM service with end-user client/load test |
| GPU operations | Driver/container validation, DCGM telemetry, runbook |
| K8s or scheduler | EKS GPU workload with scheduling, logs, and rollback |
| AI/HPC scaling | Multi-GPU NCCL on G6; multi-node EFA/NCCL only on supported P-series |
| Customer escalation | Baseline, controlled fault, evidence diagnosis, retest, recommendation |

## Lab 0: Cloud foundation and cost control

**Outcome:** protect the customer account before creating GPU cost.

```bash
aws configure sso
aws sts get-caller-identity
aws configure get region

# Discover G6 capacity instead of assuming an AZ has it.
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=g6.xlarge,g6.12xlarge \
  --query 'InstanceTypeOfferings[].{AZ:Location,Type:InstanceType}' --output table

# Find EFA-capable types in your selected Region.
aws ec2 describe-instance-types --filters Name=network-info.efa-supported,Values=true \
  --query 'InstanceTypes[].InstanceType' --output text
```

Create a monthly AWS Budget alert in the Console. Tag everything with `Project=nvidia-portfolio`, `Owner=hara`, and `AutoStop=required`. Before every lab, record expected hourly cost, stop condition, and evidence needed.

## Lab 1: G6 GPU node readiness

**Scenario:** Validate a new L4 GPU node before deploying a customer workload.

Launch `g6.xlarge` using a current AWS Deep Learning Base OSS NVIDIA Driver GPU AMI, 100-150 GB gp3, and SSH inbound only from your public IP. Use SSM/private networking in the production design.

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi topo -m
docker --version
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
python gpu_smoke_test.py
```

Save terminal output and JSON. Explain driver versus CUDA runtime versus framework compatibility, VRAM capacity, GPU utilization, and topology.

## Lab 2: Customer-facing LLM service

**Scenario:** A claims team wants an internal assistant for 25 concurrent users with a p95 latency target.

Use the existing [G6 NIM LLM PoC](../g6-nim-llm-poc/README.md). It starts an NVIDIA NIM/TensorRT-LLM-backed OpenAI-compatible endpoint, calls it as an end user, and measures concurrent load.

```bash
python client.py 'Summarize this insurance claim in three neutral bullets.'
python load_test.py --requests 20 --concurrency 1 --max-tokens 100
python load_test.py --requests 60 --concurrency 4 --max-tokens 100
python load_test.py --requests 100 --concurrency 8 --max-tokens 100
```

Record p50/p95, requests/sec, tokens/sec, GPU memory/utilization, errors, model/version, prompt/output tokens, and local AWS price. Recommend G4dn only if it meets measured SLO and memory needs; recommend G6 for L4 headroom and inference throughput.

## Lab 3: DCGM operations and observability

**Scenario:** “GPU jobs are slow.” Make the platform observable first.

```bash
docker run -d --restart unless-stopped --gpus all --name dcgm-exporter --net host \
  nvcr.io/nvidia/k8s/dcgm-exporter:latest
curl --fail http://127.0.0.1:9400/metrics | grep -E 'DCGM_FI_DEV_(GPU_UTIL|FB_USED|POWER_USAGE)' | head
```

Do not expose port 9400 publicly. Use an SSH tunnel for a lab or private Prometheus/VPC networking in production. Write a runbook for OOM, low GPU utilization, temperature/power, XID errors, GPU invisible in containers, and latency increases.

## Lab 4: EKS GPU workload

**Scenario:** A platform team needs repeatable GPU scheduling, not SSH-managed containers.

Create EKS with a temporary GPU managed node group. Start with the device plugin/default GPU support; then learn NVIDIA DRA as the advanced option.

```bash
kubectl get nodes -o wide
kubectl describe node <GPU_NODE_NAME> | grep nvidia.com/gpu
kubectl run gpu-smoke --restart=Never \
  --image=nvidia/cuda:12.4.1-base-ubuntu22.04 \
  --overrides='{"spec":{"containers":[{"name":"gpu-smoke","image":"nvidia/cuda:12.4.1-base-ubuntu22.04","command":["nvidia-smi"],"resources":{"limits":{"nvidia.com/gpu":1}}}]}}'
kubectl logs gpu-smoke
kubectl delete pod gpu-smoke
```

Document GPU requests/limits, node labels/taints, namespace quotas, logging, rollback, and isolation.

## Lab 5: Multi-GPU NCCL on one G6 node

**Scenario:** Training goes from one GPU to four and speedup is weak.

Use short-lived `g6.12xlarge` (four L4 GPUs). Copy `distributed_allreduce.py` from the Lambda package, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
torchrun --standalone --nproc_per_node=4 distributed_allreduce.py | tee g6-single-node.jsonl
nvidia-smi topo -m
```

Capture correctness plus bandwidth/latency by message size. Explain why linear speedup is not guaranteed: topology, PCIe, CPU affinity, data loading, framework and model behavior.

## Lab 6: Multi-node EFA/NCCL and AWS Batch

**Scenario:** Distributed training across nodes needs high-performance networking and reproducible scheduling.

This is advanced and budget-gated. AWS’s EFA/NCCL guidance specifies accelerated P-series instances, not G6. Use matching P-series nodes, cluster placement group, self-referencing security group, EFA interfaces, supported AMIs, and the official EFA/NCCL validation path.

```bash
fi_info -p efa 2>/dev/null || true
ls -l /dev/infiniband
ibstat
```

Implement the workload as an AWS Batch multi-node parallel job after proving the infrastructure. Keep container/image digest, data path, job definition, retry behavior, and results versioned.

## Lab 7: Customer escalation drill

**Scenario:** “An 8-node training job has poor scaling. Fix it before tomorrow.”

1. Establish workload, baseline, target scale, data, framework, SLO, and changes.
2. Verify node health, CUDA/driver/framework versions, GPU topology, DCGM evidence, and errors.
3. Compare one GPU, one node/multi-GPU, then multi-node measurements.
4. Validate EFA/RDMA bandwidth and latency before altering NCCL variables.
5. Check NIC selection, placement group, security group, CPU/NUMA affinity, shared memory, data loading, and contention.
6. Change one hypothesis, measure, document, then recommend a safe action/rollback.

## Portfolio and exam evidence

- Repository: reproducible commands, code, redacted architecture, outputs.
- Benchmark table per lab—no unsupported performance claims.
- One fault/fix/retest runbook.
- One customer readout: requirements, architecture, risks, cost/capacity, evidence, recommendation.
- Five-minute executive pitch and 15-minute technical workshop.

Use the labs to prove GPU topology, Linux/containers, NCCL, DCGM, EKS/Slurm, EFA/RDMA concepts, capacity planning, security, reliability, and customer performance triage. This is the practical material that makes an AI infrastructure/networking professional exam credible in an NVIDIA interview.
