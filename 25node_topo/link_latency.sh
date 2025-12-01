#!/bin/bash

echo "Configuring clab-dual-bridge-topology-host1 to host10..."

for i in $(seq 13 25); do
  HOST="clab-century-serf$i"
  IP_SUFFIX=$((10 + i))
  IP="10.0.2.$IP_SUFFIX"

  #echo "[$HOST] Setting up eth1 with IP $IP/24"

  # Apply 10ms delay
  containerlab tools netem set -n "$HOST" -i eth1 --delay 10ms
  
  echo "[$HOST] Configuration done."
done


# Apply 100ms delay on the link facing br_left
sudo tc qdisc add dev eth20 root netem delay 50ms
 
# Apply 100ms delay on the link facing br_right (optional, if you want symmetric delay)
sudo tc qdisc add dev eth27 root netem delay 50ms
