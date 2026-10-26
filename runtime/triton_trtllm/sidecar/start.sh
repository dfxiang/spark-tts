#!/usr/bin/env bash
set -euo pipefail

model_dir="${MODEL_DIR:-/models/Spark-TTS-0.5B}"
artifact_dir="${ARTIFACT_DIR:-/artifacts}"
checkpoint_dir="${artifact_dir}/tllm_checkpoint_bf16"
engine_dir="${artifact_dir}/trt_engines_bf16"
model_repo="/model_repo"
runtime_dir="/workspace/runtime/triton_trtllm"

mkdir -p "${artifact_dir}"

if [[ ! -f "${checkpoint_dir}/config.json" ]]; then
    echo "Converting Spark-TTS LLM to TensorRT-LLM BF16 checkpoint..."
    rm -rf "${checkpoint_dir}"
    python3 "${runtime_dir}/scripts/convert_checkpoint.py" \
        --model_dir "${model_dir}/LLM" \
        --output_dir "${checkpoint_dir}" \
        --dtype bfloat16
fi

if [[ ! -f "${engine_dir}/config.json" ]]; then
    echo "Building TensorRT-LLM BF16 engine..."
    rm -rf "${engine_dir}"
    trtllm-build \
        --checkpoint_dir "${checkpoint_dir}" \
        --output_dir "${engine_dir}" \
        --max_batch_size 1 \
        --max_input_len 1024 \
        --max_seq_len 4096 \
        --max_num_tokens 4096 \
        --gemm_plugin bfloat16
fi

rm -rf "${model_repo}"
cp -r "${runtime_dir}/model_repo" "${model_repo}"
cp -r "${runtime_dir}/sidecar/configs/." "${model_repo}/"

tritonserver \
    --model-repository="${model_repo}" \
    --model-load-thread-count=1 \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002 &
triton_pid=$!

cleanup() {
    kill "${triton_pid}" 2>/dev/null || true
    wait "${triton_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
    if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v2/health/ready', timeout=2)" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${triton_pid}" 2>/dev/null; then
        wait "${triton_pid}"
        exit $?
    fi
    sleep 1
done

python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v2/health/ready', timeout=2)"

export SPARK_TTS_TRITON_URL=http://127.0.0.1:8000
python3 -m uvicorn runtime.triton_trtllm.sidecar.adapter:app \
    --host 0.0.0.0 \
    --port 8003
