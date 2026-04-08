from __future__ import annotations

import argparse
from pathlib import Path
import sys

from acoustic_modem.config import DEFAULT_CONFIG
from acoustic_modem.framing import build_frame, bytes_to_bits, validate_text
from acoustic_modem.tx import synthesize_transmission, write_wav
from acoustic_modem.types import FramingError


EXIT_SUCCESS = 0
EXIT_INVALID_INPUT = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m acoustic_modem.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tx_parser = subparsers.add_parser("tx", help="Encode text to a transmit WAV")
    tx_parser.add_argument("--text", required=True, help="Printable ASCII payload")
    tx_parser.add_argument("--out", required=True, type=Path, help="Output WAV path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "tx":
        return _run_tx(args)

    parser.error(f"unsupported command: {args.command}")
    return EXIT_INVALID_INPUT


def _run_tx(args: argparse.Namespace) -> int:
    cfg = DEFAULT_CONFIG

    try:
        payload = validate_text(args.text, cfg)
        frame = build_frame(payload, cfg)
        frame_bits = bytes_to_bits(frame)
        samples = synthesize_transmission(frame_bits, cfg)
        write_wav(args.out, samples, cfg.sample_rate_hz)
    except (FramingError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    duration_seconds = samples.size / cfg.sample_rate_hz
    print(
        f"length={len(payload)} path={args.out} "
        f"sample_rate={cfg.sample_rate_hz} duration={duration_seconds:.2f}s"
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
