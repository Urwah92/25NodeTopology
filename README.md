# Updated 5-Node Topology with Serf, Containerlab, K3s, Custom Scheduler, and Controller DaemonSet

This repository provides a complete 5-node Kubernetes-based topology using Containerlab. Each node is configured with:

- Serf for decentralized cluster membership
- K3s as the lightweight Kubernetes distribution
- A custom Kubernetes scheduler
- A controller DaemonSet that exposes real-time node resource metrics

## Files Included

- `ceos5node.yaml`: Main topology file defining nodes and configuration
- `Dockerfile`: Builds the base image used by each node
- `test-pod.yaml`: Test pod manifest for verifying the custom scheduler

## Components

### Custom Scheduler

- Makes pod scheduling decisions based on actual node resource availability
- Pulls metrics from the `resource-api` container running in the controller DaemonSet
- Considers CPU usage, memory usage, PSI (Pressure Stall Information), and available cores

### Controller DaemonSet

- Runs on every node in the topology
- Continuously monitors system-level resource usage
- Outputs current resource offers to `/tmp/resource_offer.json`
- Exposes an HTTP API on port 8080 for the scheduler to query node status

### Liqo Integration

- Liqo installation is initiated through the `ceos5node.yaml` file
- Currently under development — some issues may prevent successful installation
- Manual updates may be required to complete Liqo federation

## For deployment follow information below.

# 5 Nodes Topology with Serf & Containerlab

This project sets up a 5-node network topology using [Containerlab](https://containerlab.dev), along with Serf agents for decentralized cluster membership and node communication. It's useful for experimenting with service discovery, failure detection, and distributed coordination in lab environments.

---

## 🗂️ Project Structure

- `ceso5node.yml` — Main Containerlab topology definition file.
- `clab-century/` — Contains node-specific configs, TLS materials, inventory, and logs.
- `setup_nodes.sh` — Orchestrates the setup of nodes.
- `ipaddressing.sh` — Assigns IP addresses to nodes.
- `serf_agents_start.sh` — Starts Serf agents.
- `serf_agents_joining.sh` — Manages the joining of nodes to the Serf cluster.
- `serf/` —  Contains Serf binary .
- `pytogoapi/` —  Web Server API for Python (custom implementation).

---

## 🚀 Getting Started

### Prerequisites

- Docker & Containerlab installed
- Bash or Linux shell
- Git (for cloning and version control)

### Setup Steps

```bash
# Step 1: Clone the repo
git clone https://github.com/abmuslim/gopyserf.git

# Step 2: Create the topology
python3 generate_topology.py --nodes=5

# Step 3: Deploy the topology
sudo containerlab deploy -t century_clab.yaml

# Step 4: Start Serf agents
./serf_agents_start.sh

# Step 5: Join nodes into the cluster
./serf_agents_joining.sh
