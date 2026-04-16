#!/usr/bin/env bash

./buyer \
  -rpc-addr 127.0.0.1:7373 \
  -event buyer.request \
  -ifname eth0 \
  -http-host 0.0.0.0 -http-port 8090 \
  -arrival-lambda-per-hour 0.5 \
  -app-id 0 \
  -app1-pct 34 -app2-pct 33 -app3-pct 33 \
  -score-min 0.0 -score-max 1.0 \
  -budget-min 0.0 -budget-max 3.0
