I created a precise AWS-first practice roadmap tailored to both NVIDIA job descriptions:
[AWS NVIDIA job labs guide](\outputs\\aws-nvidia-job-labs\\AWS_NVIDIA_JOB_LABS.md)
[GPU readiness Python test](\outputs\\aws-nvidia-job-labs\\gpu_smoke_test.py)
Practice in this order:
G6 node readiness: L4 GPU, CUDA, Docker, GPU topology, Python test.
G6 LLM service: NVIDIA NIM/TensorRT-LLM endpoint, concurrent users, p50/p95, cost/SLO recommendation.
DCGM observability: GPU utilization, memory, power, health, troubleshooting runbook.
EKS GPU scheduling: GPU resources, device plugin/DRA, workload deployment, logs, rollback.
G6 multi-GPU NCCL: four-GPU g6.12xlarge, all-reduce correctness and bandwidth.
P-series EFA/NCCL: true multi-node HPC networking, AWS Batch/Slurm-style scheduling, EFA validation.
Customer escalation drill: diagnose poor scale-out with evidence rather than guessing.
The critical interview distinction is:
G6: excellent for L4 inference, Triton/NIM, Kubernetes GPU workloads, DCGM, and single-node multi-GPU practice.
P-series + EFA: use this for authentic multi-node NCCL/RDMA/HPC training practice. AWS’s current EFA/NCCL guide specifies P-series accelerated instances for that path. AWS EFA/NCCL guidance
For Kubernetes, EKS supports NVIDIA GPU device management using either the traditional device plugin or the newer DRA approach; begin with the device plugin, then study DRA to discuss modern GPU allocation architecture in interviews. EKS NVIDIA GPU management
Start with Lab 1 this week. Do not launch EKS, a multi-GPU G6, or P-series until you have a written lab objective, planned stop time, and expected proof artifact.
