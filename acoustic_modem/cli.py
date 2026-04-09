from __future__ import annotations

import argparse
from pathlib import Path
import sys

from acoustic_modem.config import DEFAULT_CONFIG
from acoustic_modem.framing import build_frame, bytes_to_bits, validate_text
from acoustic_modem.reporting import (
    decode_result_summary,
    failure_reason,
    is_frame_validation_failure,
    is_invalid_input_failure,
    write_debug_artifacts,
    write_summary_json,
)
from acoustic_modem.rx import decode_wav
from acoustic_modem.tx import synthesize_transmission, write_wav
from acoustic_modem.types import DecodeResult, FailureCode, FramingError


EXIT_SUCCESS = 0
EXIT_INVALID_INPUT = 2
EXIT_SYNC_FAILED = 3
EXIT_FRAME_VALIDATION_FAILED = 4
EXIT_EXPECTATION_MISMATCH = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m acoustic_modem.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tx_parser = subparsers.add_parser("tx", help="Encode text to a transmit WAV")
    tx_parser.add_argument("--text", required=True, help="Printable ASCII payload")
    tx_parser.add_argument("--out", required=True, type=Path, help="Output WAV path")

    rx_parser = subparsers.add_parser("rx", help="Decode a recording WAV")
    rx_parser.add_argument("--in", dest="input_path", required=True, type=Path, help="Input WAV path")
    rx_parser.add_argument("--json-out", type=Path, help="Write a JSON decode summary")
    rx_parser.add_argument("--expect", help="Expected decoded text")
    rx_parser.add_argument("--debug-dir", type=Path, help="Write lightweight debug artifacts")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "tx":
        return _run_tx(args)
    if args.command == "rx":
        return _run_rx(args)

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


def _run_rx(args: argparse.Namespace) -> int:
    cfg = DEFAULT_CONFIG

    try:
        expected_text = None
        if args.expect is not None:
            expected_text = validate_text(args.expect, cfg).decode("ascii")
    except (FramingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    result = decode_wav(args.input_path, cfg)
    summary = decode_result_summary(result, input_path=args.input_path)
    if expected_text is not None:
        summary["expected_text"] = expected_text
        summary["expectation_met"] = result.success and result.decoded_text == expected_text

    try:
        if args.json_out is not None:
            write_summary_json(args.json_out, summary)
        if args.debug_dir is not None:
            write_debug_artifacts(args.debug_dir, summary)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    if not result.success:
        print(_format_failure(result), file=sys.stderr)
        return _rx_exit_code(result)

    decoded_text = result.decoded_text or ""
    print(decoded_text)

    if expected_text is not None and decoded_text != expected_text:
        print(
            f"expectation_mismatch: expected {expected_text!r}, got {decoded_text!r}",
            file=sys.stderr,
        )
        return EXIT_EXPECTATION_MISMATCH

    return EXIT_SUCCESS


def _rx_exit_code(result: DecodeResult) -> int:
    if is_invalid_input_failure(result):
        return EXIT_INVALID_INPUT
    if result.failure_code == FailureCode.SYNC_NOT_FOUND:
        return EXIT_SYNC_FAILED
    if is_frame_validation_failure(result):
        return EXIT_FRAME_VALIDATION_FAILED
    return EXIT_FRAME_VALIDATION_FAILED


def _format_failure(result: DecodeResult) -> str:
    base_messages = {
        FailureCode.INVALID_INPUT: "invalid_input: invalid user input",
        FailureCode.UNSUPPORTED_FORMAT: "unsupported_format: input is not a readable WAV file",
        FailureCode.SYNC_NOT_FOUND: "sync_not_found: no valid synchronization candidate found",
        FailureCode.INVALID_LENGTH: "invalid_length: decoded frame length was out of range",
        FailureCode.CRC_MISMATCH: "crc_mismatch: frame CRC validation failed",
        FailureCode.NON_ASCII_PAYLOAD: "non_ascii_payload: decoded payload was not printable ASCII",
        FailureCode.TRUNCATED_FRAME: "truncated_frame: recording ended before the full frame was recovered",
    }
    failure_code = result.failure_code
    if failure_code is None:
        message = f"{failure_reason(result)}: decode failed"
    else:
        message = base_messages.get(failure_code, f"{failure_reason(result)}: decode failed")

    diagnostics = [
        f"sync_found={str(result.sync_found).lower()}",
        f"weak_symbols={result.weak_symbol_count}",
        f"clipping_warning={str(result.clipping_warning).lower()}",
        f"samples_per_symbol={result.samples_per_symbol}",
    ]
    if result.start_sample is not None:
        diagnostics.append(f"start_sample={result.start_sample}")
    if result.best_candidate_score is not None:
        diagnostics.append(f"best_candidate_score={result.best_candidate_score:.3f}")

    return f"{message} ({' '.join(diagnostics)})"


if __name__ == "__main__":
    raise SystemExit(main())
