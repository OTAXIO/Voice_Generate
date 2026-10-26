import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import os,sys,json,shutil,re,subprocess,time,hashlib
from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
for option in ['project','output','labels','features']:parser.add_argument('--'+option,required=True)
parser.add_argument('--holdout',default='3,10,19,23')
parser.add_argument('--gpt-original-timing',action='store_true')
args=parser.parse_args()
ROOT=Path(args.project).resolve()
OUT=Path(args.output).resolve();OUT.mkdir(parents=True,exist_ok=True)
CODE=OUT/'code';DATA=OUT/'data';CODE.mkdir(exist_ok=True)
for d in ['3-bert','4-cnhubert','5-wav32k','7-sv_cn']:(DATA/d).mkdir(parents=True,exist_ok=True)
for d in ['temp','GPT_weights','SoVITS_weights']:(OUT/d).mkdir(exist_ok=True)
env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',version='v2Pro',hz='25hz',TEMP=str(OUT/'temp'),TMP=str(OUT/'temp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NUMBA_CACHE_DIR=str(OUT/'temp/numba'),MPLCONFIGDIR=str(OUT/'temp/mpl'))
env['PATH']=str(ROOT/'runtime')+os.pathsep+env['PATH']
env['PYTHONPATH']=os.pathsep.join(map(str,[CODE,ROOT,ROOT/'GPT_SoVITS',ROOT/'GPT_SoVITS/BigVGAN',ROOT/'GPT_SoVITS/eres2net']))
os.environ.update(env)
for p in env['PYTHONPATH'].split(os.pathsep):sys.path.insert(0,p)
import numpy as np,soundfile as sf,torch,yaml,librosa
torch.set_num_threads(4)
from common import install_overlay
install_overlay(ROOT, OUT)
labels=Path(args.labels).read_text(encoding='utf-8').splitlines()
textlines={x.split('\t')[0]:x for x in (Path(args.features)/'2-name2text.txt').read_text(encoding='utf-8').splitlines()}
holdout={int(v) for v in args.holdout.split(',') if v.strip()};manifest=[];train=[]
assert len(labels)-len(holdout & set(range(len(labels))))>=2, 'Need at least two training clips'
for i,line in enumerate(labels):
    path,_,_,text=line.split('|',3);name=Path(path).name
    a,sr=sf.read(str(ROOT/path))
    a=a.mean(axis=1) if a.ndim>1 else a
    if sr!=32000:a=librosa.resample(a,orig_sr=sr,target_sr=32000)
    sr=32000;frame=int(sr*.02)
    energy=np.array([np.sqrt(np.mean(a[j:j+frame]**2)) for j in range(0,len(a),frame)])
    active=np.where(energy>10**(-40/20))[0]
    assert len(active), 'Silent clip: '+name
    start=max(0,int(active[0]*frame-.12*sr));end=min(len(a),int((active[-1]+1)*frame+.12*sr));a=a[start:end]
    energy=np.array([np.sqrt(np.mean(a[j:j+frame]**2)) for j in range(0,len(a),frame)])
    edges=np.diff(np.r_[False,energy<10**(-40/20),False].astype(int));parts=[];pos=0
    for lo,hi in zip(np.where(edges==1)[0],np.where(edges==-1)[0]):
        if lo>0 and hi<len(energy) and (hi-lo)*.02>.8:
            l=int(lo*frame);h=int(hi*frame);pad=int(.225*sr)
            parts.extend([a[pos:l+pad],a[h-pad:h]]);pos=h
    parts.append(a[pos:]);a=np.concatenate(parts).astype(np.float32)
    a=.475*a/max(np.max(np.abs(a)),1e-8)+.5*a
    dest=DATA/'5-wav32k'/name;sf.write(str(dest),a,sr)
    manifest.append(dict(id=i,file=name,text=text,seconds=len(a)/sr,split='holdout' if i in holdout else 'train'))
    if i not in holdout:
        train.append(textlines[name]);shutil.copy2(Path(args.features)/'3-bert'/(name+'.pt'),DATA/'3-bert'/(name+'.pt'))
(DATA/'2-name2text.txt').write_text('\n'.join(train)+'\n',encoding='utf-8')
(OUT/'dataset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('PREPARED',len(train),'training clips; holdout',len(holdout),flush=True)
from feature_extractor.cnhubert import CNHubert
from module.models import SynthesizerTrn
from sv import SV
hubert=CNHubert(str(ROOT/'GPT_SoVITS/pretrained_models/chinese-hubert-base')).half().cuda().eval()
sv=SV('cuda',True)
ckpt=torch.load(str(ROOT/'GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth'),map_location='cpu',weights_only=False)
hps=ckpt['config'];hps['model']['semantic_frame_rate']='25hz';hps['model']['version']='v2Pro'
vq=SynthesizerTrn(hps['data']['filter_length']//2+1,hps['train']['segment_size']//hps['data']['hop_length'],n_speakers=hps['data']['n_speakers'],**hps['model']).half().cuda().eval();vq.load_state_dict(ckpt['weight'],strict=False)
sem=['item_name\tsemantic_audio']
for row in manifest:
    if row['split']!='train':continue
    name=row['file'];a,sr=sf.read(str(DATA/'5-wav32k'/name));a16=librosa.resample(a,orig_sr=sr,target_sr=16000).astype(np.float32)
    with torch.no_grad():
        wav=torch.from_numpy(a16).unsqueeze(0).cuda().half()
        ssl=hubert.model(wav*1145.14)['last_hidden_state'].transpose(1,2)
        with (DATA/'4-cnhubert'/(name+'.pt')).open('wb') as f: torch.save(ssl.cpu(),f)
        with (DATA/'7-sv_cn'/(name+'.pt')).open('wb') as f: torch.save(sv.compute_embedding3(wav).cpu(),f)
        tokens=vq.extract_latent(ssl)[0,0].tolist();sem.append(name+'\t'+' '.join(map(str,tokens)))
(DATA/'6-name2semantic.tsv').write_text('\n'.join(sem)+'\n',encoding='utf-8')
del hubert,sv,vq,ckpt;torch.cuda.empty_cache()
s2=json.loads((ROOT/'GPT_SoVITS/configs/s2v2Pro.json').read_text(encoding='utf-8'))
s2['train'].update(epochs=12,batch_size=4,learning_rate=.00005,pretrained_s2G=str(ROOT/'GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth'),pretrained_s2D=str(ROOT/'GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth'),if_save_latest=True,if_save_every_weights=True,save_every_epoch=4,gpu_numbers='0')
(DATA/'logs_s2_v2Pro').mkdir(exist_ok=True)
s2['data']['exp_dir']=str(DATA);s2['model']['version']='v2Pro';s2.update(s2_ckpt_dir=str(OUT/'s2_logs'),save_weight_dir=str(OUT/'SoVITS_weights'),name='管理员_clean',version='v2Pro')
(OUT/'s2.json').write_text(json.dumps(s2,ensure_ascii=False,indent=2),encoding='utf-8')
s1=yaml.safe_load((ROOT/'GPT_SoVITS/configs/s1longer-v2.yaml').read_text(encoding='utf-8'))
s1['train'].update(epochs=10,batch_size=4,save_every_n_epoch=2,if_save_latest=True,if_save_every_weights=True,half_weights_save_dir=str(OUT/'GPT_weights'),exp_name='管理员_clean',if_dpo=False)
s1['optimizer'].update(lr=.0001,lr_init=.00001,lr_end=.00001,warmup_steps=10,decay_steps=100)
s1['data']['num_workers']=0;s1.update(output_dir=str(OUT/'s1_logs'),pretrained_s1=str(ROOT/'GPT_SoVITS/pretrained_models/s1v3.ckpt'),train_phoneme_path=str(DATA/'2-name2text.txt'),train_semantic_path=str(DATA/'6-name2semantic.tsv'))
(OUT/'s1.yaml').write_text(yaml.safe_dump(s1,allow_unicode=True),encoding='utf-8')

if args.gpt_original_timing:
    original=Path(args.features)/'6-name2semantic.tsv'
    if not original.is_file():raise FileNotFoundError('Original semantic features required: '+str(original))
    gpt_data=OUT/'gpt_data';(gpt_data/'3-bert').mkdir(parents=True,exist_ok=True)
    names={line.split('\t')[0] for line in train}
    for filename in ['2-name2text.txt','6-name2semantic.tsv']:
        lines=(Path(args.features)/filename).read_text(encoding='utf-8').splitlines()
        kept=[line for line in lines if line.split('\t')[0] in names or line.startswith('item_name')]
        (gpt_data/filename).write_text('\n'.join(kept)+'\n',encoding='utf-8')
    for name in names:shutil.copy2(Path(args.features)/'3-bert'/(name+'.pt'),gpt_data/'3-bert'/(name+'.pt'))
    s1.update(train_phoneme_path=str(gpt_data/'2-name2text.txt'),train_semantic_path=str(gpt_data/'6-name2semantic.tsv'))
    (OUT/'s1.yaml').write_text(yaml.safe_dump(s1,allow_unicode=True),encoding='utf-8')

print('Prepared data and configs. Run tools/train.py next.',flush=True)
