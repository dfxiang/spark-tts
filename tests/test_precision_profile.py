from pathlib import Path

from runtime.triton_trtllm.precision import get_precision_profile


def test_bf16_batch_profile_uses_independent_engine_directory() -> None:
    profile = get_precision_profile("bf16")
    artifacts = Path("/artifacts")

    assert profile.engine_dir(artifacts, max_batch_size=1) == (
        artifacts / "trt_engines_bf16"
    )
    assert profile.engine_dir(artifacts, max_batch_size=4) == (
        artifacts / "trt_engines_bf16_b4"
    )
    assert profile.convert_args == ("--dtype", "bfloat16")
    assert profile.build_args == ("--gemm_plugin", "bfloat16")
