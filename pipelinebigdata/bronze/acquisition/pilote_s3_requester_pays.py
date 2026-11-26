"""Prepare one documented Dukascopy S3 object. The default mode is offline."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path


BUCKET = 'cfg-public-proper-wallaby'
REGION = 'eu-west-1'
KEY = 'XAUUSD/2021/00/05_ticks.bi5'
MAX_BYTES = 100_000_000
ESTIMATED_USD = 0.09 * MAX_BYTES / 1_000_000_000 + 2 * 0.0004 / 1000


def download(client, output):
    directory = output.resolve(strict=True)
    target = directory / 'XAUUSD_2021-01-05_S3.bi5'
    pending = target.with_suffix('.part')
    manifest = target.with_suffix('.json')
    if any(path.exists() for path in (target, pending, manifest)):
        raise FileExistsError('Un lot pilote existe déjà.')
    request = dict(Bucket=BUCKET, Key=KEY, RequestPayer='requester')
    metadata = client.head_object(**request)
    size = metadata['ContentLength']
    storage = metadata.get('StorageClass', 'STANDARD')
    if not 0 < size <= MAX_BYTES or storage != 'STANDARD' or not metadata.get('ETag'):
        raise ValueError('Objet hors budget de taille ou hors classe Standard. GET annulé.')
    result = client.get_object(**request, IfMatch=metadata['ETag'], Range=f'bytes=0-{size - 1}')
    with closing(result['Body']) as body:
        if result['ContentLength'] != size:
            raise ValueError('Longueur GET inattendue.')
        digest = hashlib.sha256()
        count = 0
        with pending.open('xb') as stream:
            while block := body.read(min(1024 * 1024, size - count + 1)):
                count += len(block)
                if count > size:
                    raise ValueError('Réponse plus longue que prévu.')
                stream.write(block)
                digest.update(block)
    if count != size:
        raise ValueError('Réponse tronquée. Le fichier partiel reste séparé.')
    pending.rename(target)
    with manifest.open('x', encoding='utf-8') as stream:
        json.dump({'status': 'downloaded', 'validation_status': 'pending', 'source': 'Dukascopy S3 Requester Pays',
                   'region': REGION, 'bucket': BUCKET, 'key': KEY, 'storage_class': storage,
                   'bytes': count, 'sha256': digest.hexdigest(), 'etag': metadata['ETag']}, stream, indent=2)
        stream.write('\n')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-autorisee', action='store_true')
    parser.add_argument('--profile')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not args.execution_autorisee:
        print(json.dumps({'mode': 'offline', 'bucket': BUCKET, 'region': REGION, 'key': KEY,
                          'requests': '1 HEAD + at most 1 GET', 'max_bytes': MAX_BYTES,
                          'estimated_usd_before_tax_and_free_allowance': ESTIMATED_USD}, indent=2))
        return
    if not args.profile or not args.out:
        parser.error('--profile et --out sont requis pour une exécution autorisée.')
    from boto3 import Session
    from botocore.config import Config
    session = Session(profile_name=args.profile)
    client = session.client('s3', region_name=REGION, config=Config(retries={'total_max_attempts': 1}))
    print(download(client, args.out))


if __name__ == '__main__':
    main()
