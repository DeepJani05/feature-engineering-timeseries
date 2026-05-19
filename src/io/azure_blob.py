"""Azure Blob Storage I/O for parquet feature files.

Tiny on purpose. The pipeline doesn't care where files live — local
disk, Azure Blob, S3 — so we keep the interface to two functions and
import the SDK lazily.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_parquet(uri: str, *, storage_connection_string: str | None = None) -> pd.DataFrame:
    """Read a parquet file from local disk or Azure Blob.

    URI conventions:
        - ``./local/path.parquet``       -> local file
        - ``az://<container>/<blob>``    -> Azure Blob
    """
    if uri.startswith("az://"):
        return _read_blob_parquet(uri, storage_connection_string)
    return pd.read_parquet(uri)


def write_parquet(
    df: pd.DataFrame, uri: str, *, storage_connection_string: str | None = None
) -> str:
    """Write a parquet file to local disk or Azure Blob. Returns the URI written."""
    if uri.startswith("az://"):
        return _write_blob_parquet(df, uri, storage_connection_string)
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    return str(path)


# ---------------------------------------------------------- internals


def _parse_blob_uri(uri: str) -> tuple[str, str]:
    """az://container/blob/path -> ('container', 'blob/path')"""
    without_scheme = uri.removeprefix("az://")
    container, _, blob = without_scheme.partition("/")
    if not container or not blob:
        raise ValueError(f"malformed Azure Blob URI: {uri}")
    return container, blob


def _read_blob_parquet(uri: str, conn_str: str | None) -> pd.DataFrame:
    if not conn_str:
        raise ValueError("storage_connection_string is required for az:// URIs")
    from azure.storage.blob import BlobServiceClient  # lazy

    container, blob = _parse_blob_uri(uri)
    client = BlobServiceClient.from_connection_string(conn_str)
    blob_client = client.get_blob_client(container=container, blob=blob)
    buf = io.BytesIO()
    blob_client.download_blob().readinto(buf)
    buf.seek(0)
    return pd.read_parquet(buf)


def _write_blob_parquet(df: pd.DataFrame, uri: str, conn_str: str | None) -> str:
    if not conn_str:
        raise ValueError("storage_connection_string is required for az:// URIs")
    from azure.storage.blob import BlobServiceClient, ContentSettings  # lazy

    container, blob = _parse_blob_uri(uri)
    client = BlobServiceClient.from_connection_string(conn_str)
    blob_client = client.get_blob_client(container=container, blob=blob)

    buf = io.BytesIO()
    df.to_parquet(buf, index=True)
    buf.seek(0)
    blob_client.upload_blob(
        buf,
        overwrite=True,
        content_settings=ContentSettings(content_type="application/octet-stream"),
    )
    logger.info("wrote parquet to %s", uri)
    return uri
