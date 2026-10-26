import argparse

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the Spark-TTS model from Hugging Face."
    )
    parser.add_argument(
        "local_dir",
        help="Local directory in which to save the downloaded model (required).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir=args.local_dir)


if __name__ == "__main__":
    main()

