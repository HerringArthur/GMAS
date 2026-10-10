#!/usr/bin/env bash
# Run with the retriever Conda environment activated.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

save_path=/data/workspace/xuhaobin/xzy/data/MASsearch/search_data

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=/data/workspace/xuhaobin/xzy/data/MASsearch/e5-base-v2-real

CUDA_VISIBLE_DEVICES=1 python -u examples/search/retriever/retrieval_server.py \
  --index_path "$index_file" \
  --corpus_path "$corpus_file" \
  --topk 3 \
  --retriever_name "$retriever_name" \
  --retriever_model "$retriever_path" \
  --faiss_gpu \
  --faiss_gpu_batch_size 65536 \
  --retrieval_batch_size 64 \
  --port 8001
