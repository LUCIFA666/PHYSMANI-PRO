"""Pin immutable Orbax checkpoints before the training manager removes them."""
import hashlib
import json
import os
from pathlib import Path
import shutil


def snapshot_checkpoint(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        receipt = json.loads((destination/'our_mani_receipt.json').read_text())
        if receipt['source'] != str(source.resolve()):
            raise ValueError('snapshot destination already belongs to a different checkpoint')
        return receipt
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+'.tmp')
    if temporary.exists():
        shutil.rmtree(temporary)
    # Orbax finalized directories are immutable. Hard links survive subsequent
    # retention cleanup without copying multi-GB arrays or changing the source.
    shutil.copytree(source, temporary, copy_function=os.link)
    files = []
    for path in sorted(p for p in temporary.rglob('*') if p.is_file()):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        files.append({'path': str(path.relative_to(temporary)), 'bytes': path.stat().st_size,
                      'sha256': digest.hexdigest()})
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    receipt = {'source': str(source.resolve()), 'checkpoint_sha256': identity, 'files': files,
               'optimizer_updates': int(source.name)+1 if source.name.isdigit() else None,
               'snapshot_method': 'hardlinks_to_finalized_immutable_orbax_files'}
    (temporary/'our_mani_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    temporary.rename(destination)
    return receipt
