"""TensorRT-LLM precision profiles for Spark-TTS engine builds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrecisionProfile:
    name: str
    convert_args: tuple[str, ...]
    build_args: tuple[str, ...]

    def checkpoint_dir(self, artifact_root: Path) -> Path:
        return artifact_root / f"tllm_checkpoint_{self.name}"

    def engine_dir(
        self,
        artifact_root: Path,
        *,
        max_batch_size: int = 1,
    ) -> Path:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        suffix = "" if max_batch_size == 1 else f"_b{max_batch_size}"
        return artifact_root / f"trt_engines_{self.name}{suffix}"


_PROFILES = {
    "bf16": PrecisionProfile(
        name="bf16",
        convert_args=("--dtype", "bfloat16"),
        build_args=("--gemm_plugin", "bfloat16"),
    ),
}


def get_precision_profile(name: str | None) -> PrecisionProfile:
    normalized = (name or "bf16").strip().lower()
    try:
        return _PROFILES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILES))
        raise ValueError(
            f"unsupported TensorRT-LLM precision {name!r}; use one of: {supported}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--field",
        required=True,
        choices=("checkpoint-dir", "engine-dir", "convert-args", "build-args"),
    )
    args = parser.parse_args()
    profile = get_precision_profile(args.precision)

    if args.field == "checkpoint-dir":
        value = profile.checkpoint_dir(args.artifact_dir)
    elif args.field == "engine-dir":
        value = profile.engine_dir(
            args.artifact_dir,
            max_batch_size=args.max_batch_size,
        )
    elif args.field == "convert-args":
        value = " ".join(profile.convert_args)
    else:
        value = " ".join(profile.build_args)
    print(value)


if __name__ == "__main__":
    main()
