import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import argparse,json,time
from pathlib import Path
from common import environment

parser=argparse.ArgumentParser()
parser.add_argument('--project',required=True)
parser.add_argument('--output',required=True,help='A run directory; output WAV is generated.wav')
parser.add_argument('--profile',required=True,help='JSON containing model and request objects')
parser.add_argument('--text');parser.add_argument('--text-file')
parser.add_argument('--seed',type=int,default=42)
args=parser.parse_args()
project=Path(args.project).resolve();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
os.environ.update(environment(project,output))
for p in [project,project/'GPT_SoVITS',project/'GPT_SoVITS/BigVGAN']:sys.path.insert(0,str(p))
import torch,numpy as np,soundfile as sf,yaml
from TTS_infer_pack.TTS import TTS,TTS_Config
torch.set_num_threads(4)
profile=json.loads(Path(args.profile).read_text(encoding='utf-8'))
text=Path(args.text_file).read_text(encoding='utf-8') if args.text_file else args.text
if not text:parser.error('Provide --text or --text-file.')
p=output/'runtime_inference.yaml';p.write_text(yaml.safe_dump({'custom':profile['model']},allow_unicode=True),encoding='utf-8')
tts=TTS(TTS_Config(str(p)));start=time.perf_counter();req={**profile['request'],'text':text,'seed':args.seed}
chunks=list(tts.run(req));sr=chunks[-1][0];audio=np.concatenate([a for _,a in chunks])
sf.write(str(output/'generated.wav'),audio,sr)
(output/'generated.json').write_text(json.dumps(dict(model=profile['model'],request=req,seconds=len(audio)/sr,inference_seconds=time.perf_counter()-start),ensure_ascii=False,indent=2),encoding='utf-8')
print(str(output/'generated.wav'))
