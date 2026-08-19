import os
import re

import pandas as pd
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import URL, Engine, make_url

from f1_data.storage.storage_backend import StorageBackend

F1_DATABASE_URL_ENV = "F1_DATABASE_URL"
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _qualified_table(schema: str, table: str) -> str:
    if not _IDENT.match(schema) or not _IDENT.match(table):
        raise ValueError(f"invalid SQL identifiers: schema={schema!r}, table={table!r}")
    return f"{schema}.{table}"


def to_sqlalchemy_url(url: str | URL) -> URL:
    parsed = url if isinstance(url, URL) else make_url(url)
    if parsed.drivername in {"postgresql", "postgres"}:
        return parsed.set(drivername="postgresql+psycopg")
    return parsed


def get_storage_backend(url: str | None = None) -> "PostgresStorageBackend":
    """Return a Postgres backend. Requires ``F1_DATABASE_URL`` when ``url`` is omitted."""
    database_url = url if url is not None else os.environ.get(F1_DATABASE_URL_ENV)
    if not database_url:
        raise RuntimeError(
            f"{F1_DATABASE_URL_ENV} is not set. "
            "Example: postgresql://f1_predictor:f1_predictor@postgres:5432/f1_predictor"
        )
    return PostgresStorageBackend(database_url)


class PostgresStorageBackend(StorageBackend):
    def __init__(self, database_url: str | URL, engine: Engine | None = None):
        self.engine = engine or create_engine(to_sqlalchemy_url(database_url))

    def read(self, schema: str, table: str) -> pd.DataFrame:
        _qualified_table(schema, table)
        return pd.read_sql_table(table, con=self.engine, schema=schema)

    def write(
        self, schema: str, table: str, df: pd.DataFrame, *, season: int | None = None
    ) -> None:
        qualified = _qualified_table(schema, table)

        with self.engine.begin() as conn:
            if schema == "bronze":
                seasons = _bronze_seasons_to_replace(df, season)
                delete_stmt = text(f"DELETE FROM {qualified} WHERE season IN :seasons").bindparams(
                    bindparam("seasons", expanding=True)
                )
                conn.execute(delete_stmt, {"seasons": seasons})
            else:
                conn.execute(text(f"TRUNCATE TABLE {qualified}"))

            if not df.empty:
                df.to_sql(
                    table,
                    con=conn,
                    schema=schema,
                    if_exists="append",
                    index=False,
                    method="multi",
                )


def _bronze_seasons_to_replace(df: pd.DataFrame, season: int | None) -> list[int]:
    if season is not None:
        return [int(season)]
    if "season" not in df.columns:
        raise ValueError("bronze write requires season= or a season column")
    seasons = sorted({int(value) for value in df["season"].dropna().unique()})
    if not seasons:
        raise ValueError("bronze write requires season= or a non-empty season column")
    return seasons
