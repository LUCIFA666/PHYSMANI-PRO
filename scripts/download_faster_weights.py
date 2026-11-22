"""Resume the public pi05 base checkpoint with bounded HTTP range downloads.

Each object is pinned to the generation in the official GCS listing. Completed
files must match the publisher's MD5 before the directory becomes a usable
OpenPI cache. Existing sequential-download prefixes are reused, then verified.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import google_crc32c
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'artifacts/framework/openpi_cache/openpi-assets/checkpoints/pi05_base'
STATE = ROOT/'runtime/framework/download'
STATUS = ROOT/'artifacts/framework/faster_download_status.json'
STOP = ROOT/'runtime/framework/state/STOP'
CHUNK = 8*1024*1024
WORKERS = 24


def verified_checksum(path, row):
    crc, md5 = google_crc32c.Checksum(), hashlib.md5() if 'md5Hash' in row else None
    with path.open('rb') as file:
        for block in iter(lambda: file.read(16*1024*1024), b''):
            crc.update(block)
            if md5 is not None:
                md5.update(block)
    # GCS composite objects publish CRC32C but deliberately have no MD5.
    return (base64.b64encode(crc.digest()).decode() == row['crc32c']
            and (md5 is None or base64.b64encode(md5.digest()).decode() == row['md5Hash']))


def atomic_json(path, record):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(record, indent=2)+'\n')
    temp.replace(path)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    listing_path = ROOT/'runtime/framework/pi05_gcs_listing.json'
    if not listing_path.exists():
        subprocess.run(['curl', '-fsSL', '--retry', '5', '--retry-all-errors', '--max-time', '60',
            'https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints%2Fpi05_base%2Fparams%2F',
            '-o', str(listing_path)], check=True)
    listing = json.loads(listing_path.read_text())
    if listing.get('nextPageToken'):
        raise RuntimeError('incomplete GCS listing')
    objects = listing['items']
    expected_total = sum(int(row['size']) for row in objects)
    lock = (CACHE/'params.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    complete, partial = CACHE/'params', CACHE/'params.partial'
    if complete.exists():
        for row in objects:
            path = complete/row['name'].removeprefix('checkpoints/pi05_base/params/')
            if not path.exists() or path.stat().st_size != int(row['size']) or not verified_checksum(path, row):
                raise RuntimeError('completed cache failed publisher checksum validation')
        atomic_json(STATUS, {'created_utc': datetime.now(timezone.utc).isoformat(), 'stage': 'complete_checksum_verified',
            'completed_bytes': expected_total, 'total_bytes': expected_total, 'fraction': 1.,
            'publisher_checksums_verified': True, 'checksum_algorithms': 'CRC32C for every object and MD5 where supplied'})
        print('base checkpoint already complete and checksum verified', flush=True)
        return
    partial.mkdir(exist_ok=True)
    states, jobs = {}, []
    state_lock = threading.Lock()
    started = time.monotonic()
    for row in objects:
        relative = row['name'].removeprefix('checkpoints/pi05_base/params/')
        path = partial/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        size = int(row['size'])
        map_path = STATE/(hashlib.sha256(row['name'].encode()).hexdigest()[:16]+'.json')
        if path.exists() and path.stat().st_size == size and verified_checksum(path, row):
            done = set(range((size+CHUNK-1)//CHUNK))
        elif map_path.exists():
            old = json.loads(map_path.read_text())
            if old['generation'] != row['generation'] or old['size'] != size:
                raise RuntimeError('checkpoint download generation changed')
            done = set(old['done'])
        else:
            # Before this downloader creates sparse files, upstream fs.get
            # writes a sequential prefix. The final MD5 validates reused data.
            prefix = min(path.stat().st_size if path.exists() else 0, size)
            done = set(range(prefix//CHUNK))
        if size == 0:
            path.touch(exist_ok=True)
        item = {'row': row, 'path': path, 'map_path': map_path, 'done': done, 'size': size}
        states[row['name']] = item
        atomic_json(map_path, {'generation': row['generation'], 'size': size, 'done': sorted(done)})
        for index in range((size+CHUNK-1)//CHUNK):
            if index not in done:
                jobs.append((item, index))

    def progress(stage='downloading_generation_pinned_chunks'):
        finished = sum(sum(min(CHUNK, item['size']-i*CHUNK) for i in item['done']) for item in states.values())
        atomic_json(STATUS, {'created_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage,
            'completed_bytes': finished, 'total_bytes': expected_total, 'fraction': finished/expected_total,
            'parallel_requests': WORKERS, 'elapsed_seconds': time.monotonic()-started,
            'publisher_checksums_verified': stage == 'complete_checksum_verified'})

    def download(job):
        item, index = job
        if STOP.exists():
            raise InterruptedError('deadline STOP')
        row, path, size = item['row'], item['path'], item['size']
        start, end = index*CHUNK, min((index+1)*CHUNK, size)-1
        temporary = STATE/(item['map_path'].stem+f'.chunk{index}')
        url = 'https://storage.googleapis.com/openpi-assets/'+quote(row['name'], safe='/')+'?generation='+row['generation']
        command = ['curl', '-fsSL', '--retry', '5', '--retry-all-errors', '--retry-delay', '1', '--connect-timeout', '12',
                   '--max-time', '180', '--range', f'{start}-{end}', '-w', '%{http_code}', '-o', str(temporary), url]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while process.poll() is None:
            if STOP.exists():
                process.terminate()
                process.wait(timeout=10)
                raise InterruptedError('deadline STOP')
            time.sleep(.5)
        output, error = process.communicate()
        if process.returncode or output.decode() != '206' or temporary.stat().st_size != end-start+1:
            raise RuntimeError(f"range download failed {row['name']} chunk {index}: {output.decode()} {error.decode()[-300:]}")
        block = temporary.read_bytes()
        descriptor = os.open(path, os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            offset = 0
            while offset < len(block):
                offset += os.pwrite(descriptor, block[offset:], start+offset)
        finally:
            os.close(descriptor)
        temporary.unlink()
        with state_lock:
            item['done'].add(index)
            atomic_json(item['map_path'], {'generation': row['generation'], 'size': size, 'done': sorted(item['done'])})
            progress()
        return len(block)

    progress()
    # Round-robin objects prevents one giant file from monopolizing all slots.
    jobs.sort(key=lambda job: (job[1], job[0]['row']['name']))
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(download, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 20 == 0:
                print(f'verified HTTP chunks {index}/{len(jobs)}', flush=True)
    progress('verifying_publisher_checksums')
    for item in states.values():
        if item['path'].stat().st_size != item['size'] or not verified_checksum(item['path'], item['row']):
            # A reused upstream prefix was unverified until this final pass.
            # Discard only its chunk ledger so a retry downloads the full file.
            atomic_json(item['map_path'], {'generation': item['row']['generation'], 'size': item['size'], 'done': []})
            raise RuntimeError('publisher checksum mismatch; run again to repair '+str(item['path']))
    partial.rename(complete)
    progress('complete_checksum_verified')
    atomic_json(ROOT/'artifacts/framework/faster_base_weights_receipt.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source': 'gs://openpi-assets/checkpoints/pi05_base/params',
        'cache': str(complete), 'bytes': expected_total, 'all_publisher_checksums_verified': True,
        'objects': [{key: row[key] for key in ('name', 'generation', 'size', 'crc32c', 'md5Hash') if key in row} for row in objects]})
    print(f'publisher-verified pi05 base checkpoint: {expected_total} bytes at {complete}', flush=True)


if __name__ == '__main__':
    main()
