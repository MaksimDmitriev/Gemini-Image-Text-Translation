"""Command-line interface."""

import argparse
from pathlib import Path

from drawing_translator.pipeline import translate_path
from drawing_translator.providers import BACKENDS, create_provider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate Russian labels in technical drawings"
    )
    parser.add_argument("input", help="Russian drawing or folder of drawings")
    parser.add_argument("output", nargs="?", help="Output image or folder")
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="gemini",
        help="Detection/translation backend (default: gemini)",
    )
    parser.add_argument("--model", help="Override the backend's default model")
    parser.add_argument(
        "--verification-passes",
        type=int,
        default=1,
        help="Re-scan the rendered image for missed Cyrillic text (default: 1)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.verification_passes < 0:
        parser.error("--verification-passes must be zero or greater")
    try:
        provider = create_provider(args.backend, args.model)
        translate_path(
            Path(args.input),
            Path(args.output) if args.output else None,
            provider,
            args.verification_passes,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
