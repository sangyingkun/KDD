from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class DelimitedPreview:
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    sample_rows: tuple[dict[str, str], ...]


def inspect_delimited_file(path: Path, *, sample_limit: int = 5) -> DelimitedPreview:
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    frame = pd.read_csv(path, sep=separator, nrows=sample_limit)
    sample_rows = tuple(
        {column: _stringify_value(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    )
    return DelimitedPreview(
        columns=tuple(str(column) for column in frame.columns),
        dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        sample_rows=sample_rows,
    )


def _stringify_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)
