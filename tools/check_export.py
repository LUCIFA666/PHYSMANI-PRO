"""Verify the text-only source export, provenance and optional final manifest."""
import ast,hashlib,json,re,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'.pt','.pth','.ckpt','.safetensors','.onnx','.npz','.npy','.h5','.hdf5',
    '.pkl','.pickle','.bin','.engine','.mp4','.avi','.mov','.pem','.key','.pyc','.pyo'}
PATTERNS=[r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',r'\bgithub_pat_[A-Za-z0-9_]{30,}\b',
    r'\bAKIA[A-Z0-9]{16}\b',r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    r'https?://[^\s/:@]+:[^\s/@]+@(?:github\.com|[^\s/]+)']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    findings=[];count=0;python=0;total=0
    paths=[]
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if any(x in {'.git','__pycache__','.pytest_cache'}for x in rel.parts):continue
        if p.is_symlink():findings.append(dict(path=str(rel),reason='symlink'));continue
        if not p.is_file():continue
        paths.append(p);count+=1;total+=p.stat().st_size
        if p.suffix.lower()in FORBIDDEN or p.name=='.env'or p.name.startswith('.env.'):
            findings.append(dict(path=str(rel),reason='excluded binary/data/credential extension'))
        if rel.parts[0]in {'artifacts','runtime','data','datasets','logs','weights','checkpoints','third_party'}:
            findings.append(dict(path=str(rel),reason='excluded local runtime directory'))
        if p.stat().st_size>10*1024*1024:findings.append(dict(path=str(rel),reason='oversized file'))
        try:content=p.read_text(encoding='utf-8')
        except UnicodeError:
            findings.append(dict(path=str(rel),reason='non-UTF8 binary'));continue
        for index,pattern in enumerate(PATTERNS):
            if re.search(pattern,content):findings.append(dict(path=str(rel),reason='credential-like content',pattern_index=index))
        if p.suffix=='.py':
            python+=1
            try:ast.parse(content,filename=str(rel))
            except SyntaxError as e:findings.append(dict(path=str(rel),reason='Python syntax',line=e.lineno))
    provenance=json.loads((ROOT/'evidence/source_provenance.json').read_text())
    for item in provenance['copied_files']:
        p=ROOT/item['path']
        if not p.is_file()or sha(p)!=item['sha256']:
            findings.append(dict(path=item['path'],reason='source copy hash mismatch'))
    manifest=ROOT/'MANIFEST.json'
    if manifest.exists():
        expected=json.loads(manifest.read_text())['files']
        actual={str(p.relative_to(ROOT)):sha(p)for p in paths if p!=manifest}
        if actual!=expected:findings.append(dict(reason='final manifest differs',missing=sorted(set(expected)-set(actual)),extra=sorted(set(actual)-set(expected)),changed=[p for p in actual if p in expected and actual[p]!=expected[p]]))
    print(json.dumps(dict(passed=not findings,files=count,python_files=python,bytes=total,
        copied_files_verified=len(provenance['copied_files']),final_manifest_checked=manifest.exists(),findings=findings),ensure_ascii=False,indent=2))
    return 0 if not findings else 1

if __name__=='__main__':raise SystemExit(main())
