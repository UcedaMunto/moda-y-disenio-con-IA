#!/usr/bin/env python3
import argparse

from core.texture.engine import generate_texture_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate placeholder texture candidates")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--input", nargs="+", required=True, help="Input image paths")
    parser.add_argument("--n", type=int, default=8, help="Number of candidates")
    args = parser.parse_args()

    candidates = generate_texture_candidates(args.input, args.project_id, args.n)
    print("Generated candidates:")
    for item in candidates:
        print(item)


if __name__ == "__main__":
    main()
