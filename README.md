# Century – 25 Node K3s Cluster

This setup creates a **25-node K3s cluster** using **Containerlab** with two switches, one FRR router and twenty five Linux nodes.

---

## Topology

![Topology diagram](25node_topo/Topology.png)

---

## 🗂️ Project Structure

- `25node_update.yml` — Main Containerlab topology definition file.
- `clab-century/` — Contains node-specific configs, TLS materials, inventory, and logs.
- `orchestrate_serf.sh` — Orchestrates the setup of nodes.
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
   ```

## How to Run

1. Run the orchestration script:
   ```bash
   ./orchestrate_serf.sh
   ```

   > It will take some time to pull the Docker images and start all pods.

2. If the pods are **not running** on any node, run:
   ```bash
   for i in {1..25}; do echo "[serf$i] applying manifests..."; sudo docker exec -i clab-century-serf$i bash -lc 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml; for f in /tmp/qos-controller-daemonset.yaml /tmp/service-account.yaml /tmp/cluster-role.yaml /tmp/cluster-role-binding.yaml /tmp/deployment-scheduler.yaml /tmp/ram_price.yaml /tmp/storage_price.yaml /tmp/vcpu_price.yaml /tmp/vgpu_price.yaml; do if [ -s "$f" ]; then echo "  applying $f..."; k3s kubectl apply -f "$f" || echo "  [warn] failed $f"; fi; done'; done
   ```

3. To check if all pods are running on all nodes:
   ```bash
   for i in {1..25}; do echo -e "\n====== [serf$i] ======"; sudo docker exec -i clab-century-serf$i bash -lc 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml; k3s kubectl get pods -A -o wide --no-headers || echo "k3s not ready"'; done
   ```

---

## Running Sellers and Buyer (must have to wait untill all the pods run)

To start sellers, run from host:
```bash
./start_sellers.sh
```

to start buyer run from host:
```bash
./start_buyer
```
💡 **Alternative Option:** 
To run the buyer from inside each container for testing (inside the **buyer container**, path: `/opt/serfapp/`):
```bash
./config_buyer.sh
```
Then run the following command **inside the buyer container** (path: `/opt/serfapp/`).
If the file `service_discovery_v7.py` is not present in the container, copy it from this repository (path: `25node_topo/serfapp/`) into the container first.
```bash
python3 service_discovery_v7.py --geom-url http://172.20.20.17:4040/cluster-status --rtt-threshold-ms 12 --rpc-addr 127.0.0.1:7373 --timeout-s 8 --sort score_per_cpu --limit 30 --buyer-url http://127.0.0.1:8090/buyer --http-serve --http-host 0.0.0.0 --http-port 4041 --http-path /hilbert-output --loop --busy-secs 50
```
> ⚠️ **Note:** Update the `--geom-url` IP (`http://172.20.20.17:4040/cluster-status`) to match the IP of **serf1** (e.g., `172.20.20.XX`).


## Note

- **Liqo is not included** in this setup.
