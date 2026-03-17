#!/usr/bin/env bash
# Launch Qwen2.5-Coder-32B-Instruct (Q8) on Radeon 8060S via Vulkan
#
# Model:    33 GB Q8 → VRAM
# KV cache: ~25 GB at 100K ctx (256 KB/token × 100K)
# Total:    ~58 GB / 64 GB VRAM
#
# Embedding dim: 5120
# Architecture:  64 layers, 40 attn heads, 8 KV heads, head_dim 128

LLAMA_CLI="$HOME/llama.cpp/build/bin/llama-cli"
MODEL="$HOME/models/gguf/qwen32b-q8.gguf"

exec "$LLAMA_CLI" \
    -m "$MODEL" \
    -ngl 999 \
    --ctx-size 100000 \
    --threads 12 \
    -fa on \
    "$@"
