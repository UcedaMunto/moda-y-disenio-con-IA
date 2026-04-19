#!/usr/bin/env python3
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA training placeholder")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--project-id", required=False, default="default")
    args = parser.parse_args()

    print("Placeholder: LoRA training is not wired yet.")
    print(f"Dataset: {args.dataset}")
    print(f"Project: {args.project_id}")


if __name__ == "__main__":
    main()
