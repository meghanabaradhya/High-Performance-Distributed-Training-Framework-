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
