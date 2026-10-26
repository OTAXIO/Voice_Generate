"""Run paired TTS comparisons, local ASR and a speaker-embedding proxy."""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import argparse,json,gc,re,time,html
from pathlib import Path
from common import environment

def normalize(text):
    import cn2an
    return re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]','',cn2an.transform(text,'an2cn')).lower()

def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['project','output','spec']:parser.add_argument('--'+flag,required=True)
    args=parser.parse_args();project=Path(args.project).resolve();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    os.environ.update(environment(project,output))
    for p in [project,project/'GPT_SoVITS',project/'GPT_SoVITS/BigVGAN']:sys.path.insert(0,str(p))
    import torch,numpy as np,soundfile as sf,yaml,librosa
    from TTS_infer_pack.TTS import TTS,TTS_Config
    torch.set_num_threads(4)
    spec=json.loads(Path(args.spec).read_text(encoding='utf-8'));rows=[]
    (output/'audio').mkdir(exist_ok=True)
    config={'custom':dict(version='v2Pro',device='cuda',is_half=True,bert_base_path=str(project/'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'),cnhuhbert_base_path=str(project/'GPT_SoVITS/pretrained_models/chinese-hubert-base'))}
    first=spec['models'][0];config['custom'].update(t2s_weights_path=first['gpt'],vits_weights_path=first['sovits'])
    path=output/'runtime.yaml';path.write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8');tts=TTS(TTS_Config(str(path)))
    def embedding(p):
        a,sr=sf.read(str(p));a=a.mean(1) if a.ndim>1 else a
        if sr!=16000:a=librosa.resample(a,orig_sr=sr,target_sr=16000)
        with torch.no_grad():v=tts.sv_model.compute_embedding3(torch.from_numpy(a.astype(np.float32)).unsqueeze(0).cuda().half()).float().flatten()
        return torch.nn.functional.normalize(v,dim=0)
    speaker=spec.get('speaker_references',[])
    target=torch.nn.functional.normalize(torch.stack([embedding(p) for p in speaker]).mean(0),dim=0) if speaker else None
    for case in spec['models']:
        if (output/'STOP').exists():raise SystemExit('STOP marker encountered')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',case['name']):raise ValueError('Model names must be safe ASCII identifiers')
        tts.init_t2s_weights(case['gpt']);tts.init_vits_weights(case['sovits']);tts.set_ref_audio(spec['reference']['path'])
        for seed in spec.get('seeds',[42]):
            for i,test in enumerate(spec['tests']):
                request=dict(text=test['text'],text_lang='all_zh',prompt_lang='all_zh',ref_audio_path=spec['reference']['path'],prompt_text=spec['reference']['text'],temperature=.75,top_k=5,top_p=1,text_split_method='cut5',batch_size=1,speed_factor=1,fragment_interval=.2,seed=seed,parallel_infer=True,repetition_penalty=1.35,streaming_mode=False)
                request.update(spec.get('parameters',{}))
                torch.cuda.synchronize();start=time.perf_counter();chunks=list(tts.run(request));torch.cuda.synchronize();elapsed=time.perf_counter()-start
                audio=np.concatenate([a for _,a in chunks]);sr=chunks[-1][0]
                if not len(audio) or not np.isfinite(audio).all():raise RuntimeError('Invalid generated audio')
                dest=output/'audio'/f'{case["name"]}_{seed}_{i}.wav';sf.write(str(dest),audio,sr)
                similarity=float(torch.dot(embedding(dest),target).cpu()) if target is not None else None
                row=dict(model=case['name'],seed=seed,test=i,text=test['text'],path=str(dest),seconds=len(audio)/sr,inference_seconds=elapsed,rtf=elapsed/(len(audio)/sr),speaker_cosine=similarity)
                rows.append(row);(output/'comparison.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
                print('GENERATED',case['name'],seed,i,flush=True)
    del tts;gc.collect();torch.cuda.empty_cache()
    from funasr import AutoModel
    asr=AutoModel(model=str(project/'tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'),disable_update=True,device='cuda',disable_pbar=True)
    for row in rows:
        a,sr=sf.read(row['path']);rec=asr.generate(input=librosa.resample(a,orig_sr=sr,target_sr=16000).astype(np.float32),disable_pbar=True)[0]['text']
        expected=normalize(row['text']);actual=normalize(rec);row.update(asr=rec,edits=edit_distance(expected,actual),characters=len(expected),cer=edit_distance(expected,actual)/max(1,len(expected)))
        (output/'comparison.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=[]
    for case in spec['models']:
        group=[r for r in rows if r['model']==case['name']]
        summary.append(dict(model=case['name'],samples=len(group),cer=sum(r['edits'] for r in group)/sum(r['characters'] for r in group),speaker_cosine=float(np.mean([r['speaker_cosine'] for r in group])) if target is not None else None,median_rtf=float(np.median([r['rtf'] for r in group])),nonzero_cer_samples=sum(r['edits']>0 for r in group)))
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    cards=[]
    for r in rows:
        cards.append(f'<article><h3>{r["model"]} / seed {r["seed"]} / text {r["test"]}</h3><p>{html.escape(r["text"])}</p><audio controls preload="none" src="audio/{Path(r["path"]).name}"></audio><p>ASR: {html.escape(r["asr"])}</p><p>CER {r["cer"]:.1%}; voice cosine {r["speaker_cosine"]}; RTF {r["rtf"]:.3f}</p></article>')
    (output/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>新旧模型对照</title><style>body{font:16px system-ui;max-width:1000px;margin:30px auto;padding:20px;background:#101722;color:#e6edf7}article{padding:20px;margin:20px 0;background:#1b293b;border-radius:12px}audio{width:100%}</style><h1>新旧模型同条件对照</h1><p>自动指标不能替代自然度、音质和情感试听。</p>'+''.join(cards),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
