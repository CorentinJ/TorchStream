import hashlib
from pathlib import Path
from urllib.parse import urlparse

import requests


def download_file_cached(url: str) -> Path:
    cache_dir = Path.cwd() / "download_cache"
    cache_dir.mkdir(exist_ok=True)

    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    filename = f"{url_hash}{suffix}" if suffix else url_hash
    target = cache_dir / filename

    expected_size = None
    try:
        head_response = requests.head(url, allow_redirects=True, timeout=10)
        head_response.raise_for_status()
        content_length = head_response.headers.get("Content-Length")
        if content_length:
            expected_size = int(content_length)
    except requests.RequestException:
        expected_size = None

    if target.exists():
        current_size = target.stat().st_size
        if expected_size is None or current_size == expected_size:
            return target

    response = requests.get(url, stream=True)
    response.raise_for_status()
    if expected_size is None:
        content_length = response.headers.get("Content-Length")
        if content_length:
            expected_size = int(content_length)

    with open(target, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            out_file.write(chunk)

    final_size = target.stat().st_size
    if expected_size is not None and final_size != expected_size:
        raise IOError(
            f"Downloaded size mismatch for {url}: expected {expected_size} bytes, got {final_size} bytes"
        )

    return target
