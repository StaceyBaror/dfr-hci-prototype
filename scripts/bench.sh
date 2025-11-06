#!/usr/bin/env bash
: "${BASE_URL:=http://127.0.0.1:8000}"
python eval/latency_bench.py
