"""Download and verify a ready-to-train SKAPP dataset."""
import argparse
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from datasets import NAMES, ROOT, digest, read_manifest


def verify_install(folder, entry):
    if digest(folder / 'dataset.json') != entry['dataset_sha256']:
        raise ValueError('Dataset manifest checksum mismatch')
    manifest = read_manifest(folder)
    if manifest['dataset'] != folder.name or manifest['release'] != entry['release']:
        raise ValueError('Dataset name/version mismatch')
    for split, info in manifest['splits'].items():
        if info['file'] != f'{split}.npz' or split not in ('train', 'valid', 'test'):
            raise ValueError('Invalid split filename')
        path = folder / info['file']
        if path.stat().st_size != info['bytes'] or digest(path) != info['sha256']:
            raise ValueError(f'{split} checksum mismatch')


def install(archive, destination, entry):
    archive = Path(archive)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'{destination} already exists; it will not be overwritten.')
    print('Verifying archive...', flush=True)
    if archive.stat().st_size != entry['bytes'] or digest(archive) != entry['sha256']:
        raise ValueError('Archive checksum mismatch. Download a fresh copy.')
    expected = {f'{destination.name}/{name}' for name in ['dataset.json', 'train.npz', 'valid.npz', 'test.npz']}
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if {item.filename for item in members} != expected or len(members) != len(expected):
            raise ValueError('Unexpected archive contents')
        if sum(item.file_size for item in members) != entry['unpacked_bytes']:
            raise ValueError('Unpacked size mismatch')
        if any(item.is_dir() or stat.S_ISLNK(item.external_attr >> 16) for item in members):
            raise ValueError('Unexpected archive entry type')
        if shutil.disk_usage(destination.parent).free < entry['unpacked_bytes'] + 64 * 1024 ** 2:
            raise OSError('Insufficient free space to unpack the dataset')
        with tempfile.TemporaryDirectory(prefix='.skapp-unpack-', dir=destination.parent) as temporary:
            staged = Path(temporary) / destination.name
            staged.mkdir()
            for item in members:
                print(f'Extracting {Path(item.filename).name}...', flush=True)
                with bundle.open(item) as source, (staged / Path(item.filename).name).open('wb') as target:
                    shutil.copyfileobj(source, target, length=4 * 1024 ** 2)
            verify_install(staged, entry)
            staged.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=NAMES, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--archive', type=Path, help='Install a manually downloaded ZIP without a network request')
    args = parser.parse_args()
    catalog = json.loads(Path(__file__).with_name('datasets.json').read_text())
    entry = catalog['datasets'][args.dataset]
    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / args.dataset
    if destination.exists():
        verify_install(destination, entry)
        print(f'Dataset is already installed and verified: {destination}')
        return
    lock = root / f'.{args.dataset}-download.lock'
    with lock.open('x'):
        pass
    try:
        archive = args.archive
        if archive is None:
            if not entry.get('url'):
                raise ValueError('Public download is not enabled yet. Use --archive with the provided ZIP.')
            import gdown
            cache = root / '.downloads'
            cache.mkdir(exist_ok=True)
            archive = cache / entry['filename']
            if not archive.exists():
                required = entry['bytes'] + entry['unpacked_bytes'] + 64 * 1024 ** 2
                if shutil.disk_usage(root).free < required:
                    raise OSError('Insufficient free space for the download and extracted dataset')
                result = gdown.download(url=entry['url'], output=str(archive), fuzzy=True,
                                        resume=True, use_cookies=False)
                if result is None:
                    raise OSError('Download did not complete; rerun this command to resume.')
        install(archive, destination, entry)
    finally:
        lock.unlink()
    print(f'Ready: {destination}')


if __name__ == '__main__':
    main()
