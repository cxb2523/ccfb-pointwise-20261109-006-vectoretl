"""Small text-prep helper shared by the batched Orchestrator.

Kept separate from vector_etl/orchestrator.py so the new batching code
path doesn't need to touch (and risk) the legacy ETLOrchestrator.
"""

import pandas as pd


def _format_column_name(name):
    return ''.join(word.capitalize() for word in str(name).split('_'))


INTERNAL_COLUMNS = {"__row_index"}


def generate_concat_column(df, embed_columns):
    """Add a '__concat_final' column built from embed_columns (or all columns)."""
    if embed_columns:
        columns = list(embed_columns)
    else:
        columns = [c for c in df.columns.to_list() if c not in INTERNAL_COLUMNS]

    df = df.copy()
    df['__concat_final'] = df.apply(
        lambda row: ' ~ '.join(
            f"{_format_column_name(col)}: {row[col]}"
            for col in columns if col in df.columns and pd.notna(row[col])
        ),
        axis=1
    )
    return df
