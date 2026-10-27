"""Official Qwen ModelScope download with size and SHA-256 verification."""
import json,urllib.request,urllib.parse,hashlib,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
repo='iic/speech_timestamp_prediction-v1-16k-offline';target=ROOT/'models/fa-zh'
with urllib.request.urlopen('https://modelscope.cn/api/v1/models/'+repo+'/repo/files?Revision=master&Recursive=true',timeout=40) as response:listing=json.load(response)
files=[f for f in listing['Data']['Files'] if f['Type']=='blob' and f['Path']!='.gitattributes']
proof=[]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
for f in files:
 dest=(target/f['Path']).resolve()
 if not dest.is_relative_to(target.resolve()):raise ValueError('Invalid model path')
 dest.parent.mkdir(parents=True,exist_ok=True)
 if not (dest.is_file() and dest.stat().st_size==f['Size'] and digest(dest)==f['Sha256']):
  part=dest.with_suffix(dest.suffix+'.partial')
  for attempt in range(3):
   try:
    offset=part.stat().st_size if part.exists() else 0
    req=urllib.request.Request('https://modelscope.cn/models/'+repo+'/resolve/master/'+urllib.parse.quote(f['Path'],safe='/'),headers={'Range':'bytes='+str(offset)+'-'} if offset else {})
    with urllib.request.urlopen(req,timeout=90) as response:
     append=offset>0 and response.status==206
     with part.open('ab' if append else 'wb') as stream:
      marker=time.monotonic()
      while True:
       data=response.read(4*1024*1024)
       if not data:break
       stream.write(data)
       if time.monotonic()-marker>20:print(f['Path'],round(stream.tell()/1024**2),'MiB',flush=True);marker=time.monotonic()
    if part.stat().st_size!=f['Size'] or digest(part)!=f['Sha256']:raise ValueError('Model integrity mismatch')
    part.replace(dest);break
   except OSError:
    if attempt==2:raise
    time.sleep(2)
  else:raise ValueError('Incomplete model download')
 proof.append(dict(path=f['Path'],bytes=f['Size'],sha256=f['Sha256']))
 print('VERIFIED',f['Path'],flush=True)
(target/'download-provenance.json').write_text(json.dumps(dict(repo_id=repo,source='official Qwen ModelScope repository',files=proof,verified_sha256=True),indent=2),encoding='utf-8')
print('MODEL_READY '+str(target),flush=True)
