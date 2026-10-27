"""Export existing aligned clips to official Qwen SFT JSONL without modifying sources."""
import argparse,json,hashlib
from pathlib import Path

def export(manifest,wavs,reference,output):
    manifest,wavs,reference,output=map(Path,[manifest,wavs,reference,output])
    if output.exists():raise ValueError('Use a new output directory')
    if not reference.is_file():raise ValueError('Missing reference')
    rows=json.loads(manifest.read_text(encoding='utf-8'));splits={'train':[],'validation':[]};seen={}
    for row in rows:
        path=(wavs/row['file']).resolve()
        if not path.is_relative_to(wavs.resolve()) or not path.is_file():raise ValueError('Missing or invalid dataset audio')
        if not row['text'].strip():raise ValueError('Empty transcript')
        digest=hashlib.sha256(path.read_bytes()).hexdigest();split='validation' if row['split'] in {'test','validation','holdout','val'} else 'train'
        if digest in seen:raise ValueError('Duplicate audio; avoid training/validation leakage')
        seen[digest]=split;splits[split].append(dict(audio=str(path),text=row['text'],ref_audio=str(reference.resolve())))
    if not all(splits.values()):raise ValueError('Both training and held-out validation clips required')
    output.mkdir(parents=True)
    for split,items in splits.items():(output/(split+'_raw.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items),encoding='utf-8')
    report=dict(counts={k:len(v) for k,v in splits.items()},fixed_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),audio_sha256=seen,trained=False)
    (output/'dataset-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['manifest','wavs','reference','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();print(json.dumps(export(a.manifest,a.wavs,a.reference,a.output),ensure_ascii=False))
