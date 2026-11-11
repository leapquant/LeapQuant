#!/usr/bin/env bash
# Start a vLLM server with LeapQuant 8-bit recurrent states.  usage: bash serve.sh <model> [vllm serve options]
set -e
MODEL=$1; shift
exec vllm serve "$MODEL" \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.8 \
    "$@"
