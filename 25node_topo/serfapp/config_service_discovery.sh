#!/bin/bash

./service_discovery_v9_multidim_full_final_sh \
      --geom-url http://172.20.20.17:4040/cluster-status \
      --rtt-threshold-ms 12 \
      --rpc-addr 127.0.0.1:7373 --timeout-s 8 \
      --pct-start 0.25 --max-steps 8 \
      --sort score_per_cpu --limit 30 \
      --http-serve --http-host 0.0.0.0 --http-port 4041 --http-path /hilbert-output \
      --post-output-url http://localhost:5665/initiate_tx \
      --buyer-url http://127.0.0.1:8090/buyer

