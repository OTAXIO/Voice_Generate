"""Local Qwen3-TTS adapter, using the official Qwen3TTSModel API."""
import os,sys,json,time,argparse,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from prosody import synthesis_units,speech_tempo
from speech_text import prepare_text

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);a=parser.parse_args()
    job=json.loads(Path(a.job).read_text(encoding='utf-8-sig'))
    model_path=Path(job['model_path']).resolve()
    if not (model_path/'config.json').is_file():raise ValueError('Download the Qwen3-TTS Base checkpoint before inference')
    reference=Path(job['reference']).resolve()
    if not reference.is_file() or not job.get('reference_text','').strip():raise ValueError('Local reference audio and exact transcript required')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUTF8='1')
    import torch,numpy as np,soundfile as sf
    from qwen_tts import Qwen3TTSModel
    from audio_cleanup import read_local_text
    torch.set_num_threads(4);torch.manual_seed(int(job.get('seed',42)))
    gpu=torch.cuda.is_available();device='cuda:0' if gpu else 'cpu'
    dtype=torch.bfloat16 if gpu and torch.cuda.is_bf16_supported() else torch.float16 if gpu else torch.float32
    text=prepare_text(read_local_text(job['text_file']),job.get('pronunciation_readings'));units=synthesis_units(text,job.get('synthesis_unit','sentence'),int(job.get('max_context_chars',120)))
    if not units:raise ValueError('No narration text')
    model=Qwen3TTSModel.from_pretrained(str(model_path),device_map=device,dtype=dtype,attn_implementation='sdpa',local_files_only=True)
    prompt=model.create_voice_clone_prompt(ref_audio=str(reference),ref_text=job['reference_text'],x_vector_only_mode=False)
    parts=[];raw_parts=[];sentences=[];start=time.perf_counter();sr=None
    generation={key:job[key] for key in ['temperature','top_p','top_k','subtalker_temperature','subtalker_top_p','subtalker_top_k'] if key in job}
    for index,text in enumerate(units):
        if job.get('consistent_unit_seed',False):torch.manual_seed(int(job.get('seed',42)))
        wavs,rate=model.generate_voice_clone(text=text,language=job.get('qwen_language','Chinese'),voice_clone_prompt=prompt,max_new_tokens=int(job.get('max_new_tokens',1024)),**generation)
        if sr is not None and rate!=sr:raise ValueError('Sample rate changed')
        sr=rate;audio=np.asarray(wavs[0],dtype=np.float32).reshape(-1)
        if not len(audio) or not np.isfinite(audio).all():raise ValueError('Invalid generated audio')
        if np.max(np.abs(audio))>1:raise ValueError('Generated audio would clip')
        raw_parts.append(audio)
        processed=speech_tempo(audio,sr,float(job.get('speech_rate',1)),job.get('ffmpeg_executable'))
        parts.append(processed);sentences.append(dict(sentence=text,raw_samples=len(audio),processed_samples=len(processed),mode='off',removed_seconds=0.0,speech_rate=float(job.get('speech_rate',1))))
        if index+1<len(units):
            pause=np.zeros(round(sr*float(job.get('pause',.2))),dtype=np.float32)
            parts.append(pause);raw_parts.append(pause.copy())
        print('@@VOICE_EVENT '+json.dumps(dict(type='status',current=index+1,total=len(units)),ensure_ascii=False),flush=True)
    target=Path(job['output_file']);target.parent.mkdir(parents=True,exist_ok=True)
    audio=np.concatenate(parts);sf.write(str(target),audio,sr)
    raw=target.with_name(target.stem+'_原始.wav');sf.write(str(raw),np.concatenate(raw_parts),sr)
    meta=dict(backend='qwen3-tts',model_path=str(model_path),job=job,sentences=sentences,sample_rate=sr,seconds=len(audio)/sr,raw_seconds=sum(map(len,raw_parts))/sr,inference_seconds=time.perf_counter()-start,input_sha256=hashlib.sha256(Path(job['text_file']).read_bytes()).hexdigest(),raw_path=str(raw),output_path=str(target),voice_quality_approved=False)
    target.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('@@VOICE_EVENT '+json.dumps(dict(type='done',output=str(target),seconds=meta['seconds']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
