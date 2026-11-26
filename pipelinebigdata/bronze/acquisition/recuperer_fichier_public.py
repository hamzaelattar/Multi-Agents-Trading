"""Resume a public file transfer, then verify its published SHA-256."""

from hashlib import file_digest
from http.client import IncompleteRead
import re
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def sha256(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def download_file(url, path, size, digest):
    if path.is_file():
        if path.stat().st_size != size or sha256(path) != digest:
            raise ValueError(f'Le fichier existant diffère de la publication : {path.name}')
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    for attempt in range(8):
        position = partial.stat().st_size if partial.exists() else 0
        if position == size:
            if sha256(partial) != digest:
                raise ValueError(f'SHA-256 incorrect : {partial.name}')
            partial.rename(path)
            return
        if position > size:
            raise ValueError(f'Taille inattendue : {partial.name}')
        headers = {'Range': f'bytes={position}-'} if position else {}
        try:
            with urlopen(Request(url, headers=headers), timeout=55) as response:
                if position:
                    content_range = response.headers.get('Content-Range', '')
                    if response.status != 206 or not re.fullmatch(rf'bytes {position}-\d+/{size}', content_range):
                        raise ValueError(f'Le serveur refuse la reprise : {path.name}')
                with partial.open('ab') as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        stream.write(block)
        except HTTPError:
            raise
        except (URLError, ConnectionError, TimeoutError, IncompleteRead):
            current = partial.stat().st_size if partial.exists() else 0
            if attempt >= 2 and current <= position:
                raise
        if partial.exists() and partial.stat().st_size == size:
            if sha256(partial) != digest:
                raise ValueError(f'SHA-256 incorrect : {partial.name}')
            partial.rename(path)
            return
        sleep(min(12 * (attempt + 1), 45))
    raise OSError(f'Transfert incomplet : {path.name}')
