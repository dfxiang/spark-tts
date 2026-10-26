ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3
FROM ${BASE_IMAGE}

ENV PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY runtime/triton_trtllm/sidecar/requirements.txt /tmp/sidecar-requirements.txt
RUN python3 -m pip install -r /tmp/sidecar-requirements.txt \
    && rm /tmp/sidecar-requirements.txt

COPY sparktts /workspace/sparktts
COPY api /workspace/api
COPY runtime/triton_trtllm /workspace/runtime/triton_trtllm

RUN chmod +x /workspace/runtime/triton_trtllm/sidecar/start.sh

EXPOSE 8000 8001 8002 8003

ENTRYPOINT ["/workspace/runtime/triton_trtllm/sidecar/start.sh"]
