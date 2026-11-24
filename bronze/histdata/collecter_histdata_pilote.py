"""Submit the public HistData archive form for the separate XAUUSD pilot month."""

import argparse
from datetime import date
import hashlib
from html.parser import HTMLParser
from http.cookiejar import CookieJar
import json
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPCookieProcessor, Request, build_opener
import zipfile


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.active = False
        self.parameters = {}

    def handle_starttag(self, tag, attrs):
        fields = dict(attrs)
        if tag == 'form' and fields.get('id') == 'file_down' and fields.get('method', '').upper() == 'POST':
            self.action = fields.get('action')
            self.active = True
        if self.active and tag == 'input' and fields.get('type') == 'hidden' and fields.get('name'):
            self.parameters[fields['name']] = fields.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form':
            self.active = False


def entry(month):
    labelled = date.fromisoformat(month + '-01')
    url = f'https://www.histdata.com/download-free-forex-data/?/ascii/tick-data-quotes/xauusd/{labelled.year}/{labelled.month}'
    key = labelled.strftime('%Y%m')
    return url, key, f'HISTDATA_COM_ASCII_XAUUSD_T_{key}.zip'


def main(out, month):
    page, key, archive_name = entry(month)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / archive_name
    pending = destination.with_suffix('.part')
    manifest = destination.with_suffix('.json')
    if any(p.exists() for p in (destination, pending, manifest)):
        raise FileExistsError('Le pilote HistData existe déjà. Les fichiers existants sont conservés.')
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    with opener.open(page, timeout=25) as response:
        form = DownloadForm()
        form.feed(response.read().decode('utf-8', errors='replace'))
        source_page = response.geturl()
    if not form.action or form.parameters.get('fxpair') != 'XAUUSD' or form.parameters.get('datemonth') != key:
        raise ValueError('Le formulaire public ne confirme pas le lot XAUUSD demandé.')
    endpoint = urljoin(source_page, form.action)
    target = urlsplit(endpoint)
    if target.scheme != 'https' or target.hostname != 'www.histdata.com' or target.path != '/get.php':
        raise ValueError('Adresse de formulaire inattendue.')
    request = Request(endpoint, data=urlencode(form.parameters).encode('ascii'), headers={'Referer': source_page})
    digest = hashlib.sha256()
    total = 0
    with opener.open(request, timeout=60) as response:
        first = response.read(4)
        if first != b'PK\x03\x04':
            raise ValueError(f'Le formulaire public a renvoyé {response.status}, sans archive ZIP.')
        with pending.open('xb') as file:
            file.write(first)
            digest.update(first)
            total = len(first)
            for block in iter(lambda: response.read(1024 * 1024), b''):
                total += len(block)
                if total > 100_000_000:
                    raise ValueError('Archive au-delà de la limite du pilote.')
                digest.update(block)
                file.write(block)
    with zipfile.ZipFile(pending) as archive:
        members = archive.namelist()
        if archive.testzip() or not any(name.lower().endswith('.csv') for name in members):
            raise ValueError('Archive HistData corrompue ou sans CSV.')
    pending.rename(destination)
    with manifest.open('x', encoding='utf-8') as output:
        json.dump({'source': 'HistData', 'page': page, 'instrument': 'XAU/USD', 'month': month,
                   'status': 'downloaded', 'validation_status': 'pending', 'sha256': digest.hexdigest(),
                   'bytes': total, 'files': members, 'format': 'Generic ASCII tick, Bid/Ask and one volume field',
                   'documented_timezone': 'EST UTC-05:00 fixed'}, output, indent=2)
        output.write('\n')
    print(f'{destination.name}: {total:,} octets, SHA-256 {digest.hexdigest()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--month', required=True)
    args = parser.parse_args()
    main(args.out, args.month)
