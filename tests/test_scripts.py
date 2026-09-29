from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
import respx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from download_data import SOURCES, Source, download, main, sha256_of  # noqa: E402
from make_manifest import build_manifest  # noqa: E402


def test_sha256_of(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"abc")
    assert sha256_of(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@respx.mock
def test_download_verifies_hash(tmp_path):
    src = Source(
        "t",
        "https://raw.githubusercontent.com/x/y/main/a.json",
        "a.json",
        "Apache-2.0",
        expected_sha256="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    )
    respx.get(src.url).mock(return_value=httpx.Response(200, content=b"abc"))
    path = download(src, tmp_path)
    assert path.read_bytes() == b"abc"


@respx.mock
def test_download_rejects_bad_hash_and_http_errors(tmp_path):
    src = Source(
        "t", "https://raw.githubusercontent.com/x/y/main/b.json", "b.json", "MIT", expected_sha256="00"
    )
    respx.get(src.url).mock(return_value=httpx.Response(200, content=b"abc"))
    with pytest.raises(ValueError):
        download(src, tmp_path)
    assert not (tmp_path / "b.json").exists()
    respx.get(src.url).mock(return_value=httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        download(src, tmp_path)


@respx.mock
def test_main_reports_failures(tmp_path):
    for s in SOURCES.values():
        respx.get(s.url).mock(return_value=httpx.Response(500))
    assert main(["--dest", str(tmp_path)]) == 1
    respx.get(SOURCES["osworld"].url).mock(return_value=httpx.Response(200, content=b"{}"))
    assert main(["--only", "osworld", "--dest", str(tmp_path)]) == 0


def test_build_manifest_lists_files(tmp_path):
    repo_file = Path(__file__).resolve().parent.parent / "data" / "eval" / "tasks.json"
    text = build_manifest([repo_file], raw_dir=tmp_path / "missing")
    assert "data/eval/tasks.json" in text
    assert "manual" in text
