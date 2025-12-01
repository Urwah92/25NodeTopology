FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install required packages including certs for curl HTTPS
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    iproute2 \
    iputils-ping \
    net-tools \
    vim \
    sudo \
    kmod \
    systemd && \
    rm -rf /var/lib/apt/lists/*

# Install k3s
RUN curl -fsSL -o /usr/local/bin/k3s https://github.com/k3s-io/k3s/releases/download/v1.32.4+k3s1/k3s && \
    chmod +x /usr/local/bin/k3s

# Copy Kubernetes manifests
COPY qos-controller-daemonset.yaml /tmp/qos-controller-daemonset.yaml
COPY deployment-scheduler.yaml /tmp/deployment-scheduler.yaml
COPY cluster-role.yaml /tmp/cluster-role.yaml
COPY cluster-role-binding.yaml /tmp/cluster-role-binding.yaml
COPY service-account.yaml /tmp/service-account.yaml
COPY test-pod.yaml /tmp/test-pod.yaml

CMD ["sleep", "infinity"]

