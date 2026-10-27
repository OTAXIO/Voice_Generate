"""Desktop worker: direct local TTS inference, no HTTP server or browser."""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import argparse,json,time,hashlib,subprocess
from pathlib import Path
from common import environment

def emit(kind,**data):print('@@VOICE_EVENT '+json.dumps(dict(type=kind,**data),ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
    job=json.loads(Path(args.job).read_text(encoding='utf-8'))
    project=Path(job['project']).resolve();run=Path(args.job).resolve().parent
    os.environ.update(environment(project,run))
    for path in [project,project/'GPT_SoVITS',project/'GPT_SoVITS/BigVGAN']:sys.path.insert(0,str(path))
    import torch,numpy as np,soundfile as sf,yaml
    from audio_cleanup import read_local_text,split_sentences,clean_phrase_end
    from TTS_infer_pack.TTS import TTS,TTS_Config
    torch.set_num_threads(4)
    from prosody import synthesis_units,sampling
    text=read_local_text(job['text_file']);sentences=synthesis_units(text,job.get('synthesis_unit','sentence'),int(job.get('max_context_chars',120)))
    generation=sampling(job)
    if not sentences:raise ValueError('TXT 中没有可生成的文字。')
    reference_path=job['reference']
    if Path(reference_path).suffix.lower() not in ['.wav','.flac']:
        converted=run/'reference.wav'
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        completed=subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',reference_path,'-ac','1','-ar','32000',str(converted)],capture_output=True,creationflags=flags)
        if completed.returncode:raise ValueError('参考音频无法解码，请尝试 WAV 或 FLAC。')
        reference_path=str(converted)
    reference=sf.info(reference_path)
    if not 3<=reference.duration<=10:raise ValueError('参考音频须为 3～10 秒，建议使用清晰、收尾干净的完整台词。')
    model=dict(version='v2Pro',device='cuda' if torch.cuda.is_available() else 'cpu',is_half=torch.cuda.is_available(),t2s_weights_path=job['gpt'],vits_weights_path=job['sovits'],bert_base_path=str(project/'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'),cnhuhbert_base_path=str(project/'GPT_SoVITS/pretrained_models/chinese-hubert-base'))
    config=run/'inference.yaml';config.write_text(yaml.safe_dump({'custom':model},allow_unicode=True),encoding='utf-8')
    emit('status',message='正在加载本地模型，首次生成需要稍等……',total=len(sentences),current=0)
    tts=TTS(TTS_Config(str(config)));raw_parts=[];clean_parts=[];reports=[];sr=None;start=time.perf_counter()
    for index,sentence in enumerate(sentences):
        emit('status',message=f'正在生成第 {index+1}/{len(sentences)} 句',current=index,total=len(sentences))
        request=dict(text=sentence,text_lang=job.get('language','all_zh'),ref_audio_path=reference_path,prompt_text=job['reference_text'],prompt_lang=job.get('reference_language','all_zh'),**generation,text_split_method='cut0',batch_size=1,speed_factor=float(job.get('speed',1.05)),fragment_interval=0,seed=int(job.get('seed',42))+index,parallel_infer=True,streaming_mode=False)
        chunks=list(tts.run(request));sr=chunks[-1][0]
        if any(rate!=sr for rate,_ in chunks):raise ValueError('生成片段采样率不一致。')
        audio=np.concatenate([a for _,a in chunks]).astype(np.float32)/32768
        if not len(audio) or not np.isfinite(audio).all():raise ValueError('模型未返回有效音频。')
        cleaned,report=clean_phrase_end(audio,sr,job.get('cleanup','standard'))
        raw_parts.append(audio);clean_parts.append(cleaned)
        reports.append(dict(sentence=sentence,seed=request['seed'],raw_samples=len(audio),processed_samples=len(cleaned),**report))
        if index+1<len(sentences):
            gap=np.zeros(int(sr*float(job.get('pause',.18))),dtype=np.float32)
            raw_parts.append(gap);clean_parts.append(gap)
        emit('status',message=f'第 {index+1} 句已完成',current=index+1,total=len(sentences))
    target=Path(job['output_file']);target.parent.mkdir(parents=True,exist_ok=True)
    raw_target=target.with_name(target.stem+'_原始.wav')
    sf.write(str(target),np.concatenate(clean_parts),sr);sf.write(str(raw_target),np.concatenate(raw_parts),sr)
    metadata=dict(job=job,sentences=reports,sample_rate=sr,seconds=sum(map(len,clean_parts))/sr,raw_seconds=sum(map(len,raw_parts))/sr,inference_seconds=time.perf_counter()-start,input_sha256=hashlib.sha256(Path(job['text_file']).read_bytes()).hexdigest(),raw_path=str(raw_target),output_path=str(target))
    target.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    emit('done',output=str(target),raw=str(raw_target),seconds=metadata['seconds'],removed_seconds=metadata['raw_seconds']-metadata['seconds'])

if __name__=='__main__':
    try:main()
    except Exception as error:
        emit('error',message=str(error));raise
