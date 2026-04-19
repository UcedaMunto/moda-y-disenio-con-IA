#!/usr/bin/env python3
import argparse

from core.rendering.blender_runner import apply_texture_and_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply texture to model (placeholder)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--texture", required=True)
    parser.add_argument("--output", default="data/exports/output.glb")
    args = parser.parse_args()

    output = apply_texture_and_export(args.model, args.texture, args.output)
    print(f"Export generated at: {output}")


if __name__ == "__main__":
    main()
