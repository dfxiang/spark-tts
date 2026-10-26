#!/usr/bin/env bash
set -euo pipefail

model_dir="${MODEL_DIR:-/models/Spark-TTS-0.5B}"
artifact_dir="${ARTIFACT_DIR:-/artifacts}"
precision="${SPARK_TTS_TRT_PRECISION:-bf16}"
max_batch_size="${SPARK_TTS_TRT_MAX_BATCH_SIZE:-4}"
model_repo="/model_repo_llm"
runtime_dir="/workspace/runtime/triton_trtllm"

precision_field() {
    python3 -m runtime.triton_trtllm.precision \
        --precision "${precision}" \
        --artifact-dir "${artifact_dir}" \
        --max-batch-size "${max_batch_size}" \
        --field "$1"
}

checkpoint_dir="$(precision_field checkpoint-dir)"
engine_dir="$(precision_field engine-dir)"
max_num_tokens=$((max_batch_size * 4096))
max_tokens_in_kv_cache="${max_num_tokens}"
if [[ "${max_batch_size}" -gt 1 ]]; then
    max_queue_delay_us=2000
else
    max_queue_delay_us=0
fi

if [[ ! -f "${checkpoint_dir}/config.json" ]]; then
    echo "Converting missing TensorRT-LLM ${precision} checkpoint..."
    rm -rf "${checkpoint_dir}"
    read -r -a convert_args <<< "$(precision_field convert-args)"
    python3 "${runtime_dir}/scripts/convert_checkpoint.py" \
        --model_dir "${model_dir}/LLM" \
        --output_dir "${checkpoint_dir}" \
        "${convert_args[@]}"
fi

if [[ ! -f "${engine_dir}/config.json" ]]; then
    echo "Building TensorRT-LLM ${precision} batch=${max_batch_size} engine..."
    rm -rf "${engine_dir}"
    read -r -a build_args <<< "$(precision_field build-args)"
    trtllm-build \
        --checkpoint_dir "${checkpoint_dir}" \
        --output_dir "${engine_dir}" \
        --max_batch_size "${max_batch_size}" \
        --max_input_len 1024 \
        --max_seq_len 4096 \
        --max_num_tokens "${max_num_tokens}" \
        "${build_args[@]}"
fi

rm -rf "${model_repo}"
mkdir -p "${model_repo}/tensorrt_llm/1"
cp "${runtime_dir}/sidecar/configs/tensorrt_llm/config.pbtxt" \
    "${model_repo}/tensorrt_llm/config.pbtxt"
sed -i "s|\${TRT_ENGINE_DIR}|${engine_dir}|g" \
    "${model_repo}/tensorrt_llm/config.pbtxt"
sed -i \
    -e "s|\${TRT_MAX_BATCH_SIZE}|${max_batch_size}|g" \
    -e "s|\${TRT_MAX_QUEUE_DELAY_US}|${max_queue_delay_us}|g" \
    -e "s|\${TRT_MAX_TOKENS_IN_KV_CACHE}|${max_tokens_in_kv_cache}|g" \
    "${model_repo}/tensorrt_llm/config.pbtxt"

tritonserver \
    --model-repository="${model_repo}" \
    --model-load-thread-count=1 \
    --http-port=8100 \
    --grpc-port=8101 \
    --metrics-port=8102 &
triton_pid=$!

cleanup() {
    kill "${triton_pid}" 2>/dev/null || true
    wait "${triton_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
    if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/v2/health/ready', timeout=2)" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${triton_pid}" 2>/dev/null; then
        wait "${triton_pid}"
        exit $?
    fi
    sleep 1
done

python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/v2/health/ready', timeout=2)"

export SPARK_TTS_TRTLLM_URL=http://127.0.0.1:8100
python3 -m uvicorn runtime.triton_trtllm.sidecar.unified_app:app \
    --host 0.0.0.0 \
    --port 8000
