from __future__ import annotations

import json
from pathlib import Path

from acoustic_modem.types import DecodeResult, FailureCode


_FRAME_VALIDATION_FAILURES = frozenset(
    {
        FailureCode.CRC_MISMATCH,
        FailureCode.INVALID_LENGTH,
        FailureCode.NON_ASCII_PAYLOAD,
        FailureCode.TRUNCATED_FRAME,
    }
)
_INVALID_INPUT_FAILURES = frozenset({FailureCode.INVALID_INPUT, FailureCode.UNSUPPORTED_FORMAT})


def decode_result_summary(result: DecodeResult, input_path: Path | None = None) -> dict[str, object]:
    warnings: list[str] = []
    if result.clipping_warning:
        warnings.append("clipping_warning")
    if result.weak_symbol_count > 0:
        warnings.append("weak_symbols")

    return {
        "input_path": str(input_path) if input_path is not None else None,
        "success": result.success,
        "decoded_text": result.decoded_text,
        "failure_code": result.failure_code.value if result.failure_code is not None else None,
        "failure_reason": None if result.success else failure_reason(result),
        "recovered_length": result.recovered_length,
        "crc_ok": result.crc_ok,
        "frame_validation_failed": is_frame_validation_failure(result),
        "weak_symbol_count": result.weak_symbol_count,
        "samples_per_symbol": result.samples_per_symbol,
        "clipping_warning": result.clipping_warning,
        "sync_found": result.sync_found,
        "start_sample": result.start_sample,
        "best_candidate_score": result.best_candidate_score,
        "warnings": warnings,
    }


def failure_reason(result: DecodeResult) -> str:
    if result.failure_code is None:
        return "success"
    return result.failure_code.value


def is_frame_validation_failure(result: DecodeResult) -> bool:
    return result.failure_code in _FRAME_VALIDATION_FAILURES


def is_invalid_input_failure(result: DecodeResult) -> bool:
    return result.failure_code in _INVALID_INPUT_FAILURES


def write_summary_json(path: Path, summary: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def write_debug_artifacts(debug_dir: Path, summary: dict[str, object]) -> Path:
    output_dir = Path(debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "decode_summary.json"
    write_summary_json(summary_path, summary)
    return summary_path
