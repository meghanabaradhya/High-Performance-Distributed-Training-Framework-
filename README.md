# High-Performance Distributed Training Framework on Linux-Based HPC Clusters

This project demonstrates how to set up a distributed deep learning training pipeline using MPI/NCCL across multiple nodes in a high-performance computing (HPC) cluster. The guide uses Google Cloud Platform (GCP) but can be adapted for AWS, Azure, or on-prem clusters.

## Table of Contents
- [Step 1: Set Up the Cluster Infrastructure](#step-1-set-up-the-cluster-infrastructure)
- [Step 2: Install Dependencies on Each Node](#step-2-install-dependencies-on-each-node)
- [Step 3: Implement Distributed Training with MPI/NCCL](#step-3-implement-distributed-training-with-mpincc)
- [Step 4: Run Training Across Multiple Nodes](#step-4-run-training-across-multiple-nodes)
- [Step 5: Optimize Performance](#step-5-optimize-performance)
- [Step 6: Monitor & Benchmark Training](#step-6-monitor-benchmark-training)
- [Step 7: Automate Deployment with Terraform](#step-7-automate-deployment-with-terraform)
- [Expected Outcomes](#expected-outcomes)
- [Next Steps](#next-steps)

---

## Step 1: Set Up the Cluster Infrastructure

Set up a multi-node HPC cluster with GPUs/TPUs on GCP.

### 1.1 Enable Compute Engine & TPUs on GCP
```bash
gcloud services enable compute.googleapis.com tpu.googleapis.com
```
### 1.2 Create Multi-GPU Virtual Machines
```bash
gcloud compute instances create gpu-node-1 gpu-node-2 \
  --machine-type n1-standard-8 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --image-family=deep-learning-pytorch \
  --image-project=deeplearning-platform-release \
  --zone=us-central1-a
```

Each VM will have an NVIDIA T4 GPU for distributed training.
### Step 2: Install Dependencies on Each Node
2.1 Install Required Packages
Run the following commands on each node:
```bash
sudo apt update && sudo apt install -y openmpi-bin libopenmpi-dev
pip install torch torchvision torchaudio horovod mpi4py

```
2.2 Enable SSH for Inter-Node Communication
```bash
gcloud compute config-ssh
```
Now you can SSH into any node:
```bash
ssh gpu-node-1
```
### Step 3: Implement Distributed Training with MPI/NCCL
This script uses MPI + Horovod for distributed training across multiple GPUs.
### Step 4: Run Training Across Multiple Nodes
Use mpirun to execute the training across multiple nodes:
```
mpirun --allow-run-as-root -np 2 -H gpu-node-1,gpu-node-2 \
  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  python train.py
```
This command will distribute the training across gpu-node-1 and gpu-node-2.
###Step 5: Optimize Performance
5.1 Enable NCCL for Fast GPU Communication
```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
```
5.2 Use Mixed Precision for Speed Gains
Modify train.py to use FP16 precision with Apex AMP:
```python
from apex import amp
model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
```
5.3 Tune Batch Size and Learning Rate
```bash
python train.py --batch-size 256 --lr 0.001
```
### Step 6: Monitor & Benchmark Training
6.1 Use NVIDIA Profiling Tools
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
This tracks GPU usage in real-time.
```

6.2 Compare Single-Node vs. Multi-Node Performance
```bash
mpirun -np 1 python train.py  # Single-node training
mpirun -np 2 python train.py  # Multi-node training
```
Step 7: Automate Deployment with Terraform
7.1 Install Terraform
```bash
sudo apt install terraform
```
7.2 Define Terraform Script (main.tf)

```main.tf
resource "google_compute_instance" "gpu_nodes" {
  count = 2
  name  = "gpu-node-${count.index}"
  machine_type = "n1-standard-8"
  boot_disk { initialize_params { image = "deeplearning-platform-release" } }
  network_interface { network = "default" }
}
```
7.3 Deploy with Terraform
```bash
terraform init
terraform apply -auto-approve
```
This will automatically provision and configure multi-node GPU instances.

### Expected Outcomes
10x faster AI training using multi-GPU parallelism.
Reduced GPU idle time with optimized NCCL/MPI communication.
Fully automated training pipeline with Terraform.




