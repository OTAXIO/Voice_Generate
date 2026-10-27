"""Optional bounded context, keeping defaults and complete words/sentences intact."""
import re

def synthesis_units(text,mode='sentence',max_chars=120):
    if mode not in {'sentence','paragraph'}:raise ValueError('Unknown synthesis unit')
    if not 30<=max_chars<=300:raise ValueError('Context limit must be 30..300 characters')
    def split(part):return [p.strip() for p in re.split(r'(?<=[。！？!?])\s*|(?<=\.)\s+|\n+',part) if p.strip()]
    if mode=='sentence':return split(text)
    result=[]
    for paragraph in re.split(r'\n\s*\n',text.strip()):
        current=''
        for sentence in split(paragraph):
            combined=(current+' '+sentence).strip()
            if current and len(combined)>max_chars:result.append(current);current=sentence
            else:current=combined
        if current:result.append(current)
    return result

def sampling(job):
    values=dict(temperature=float(job.get('temperature',.75)),top_k=int(job.get('top_k',5)),top_p=float(job.get('top_p',1)),repetition_penalty=float(job.get('repetition_penalty',1.35)))
    if not 0<values['temperature']<=2 or not 1<=values['top_k']<=100 or not 0<values['top_p']<=1 or not 1<=values['repetition_penalty']<=3:raise ValueError('Invalid sampling settings')
    return values


def speech_tempo(audio,sr,factor,ffmpeg):
    import numpy as np,subprocess,os
    from pathlib import Path
    a=np.asarray(audio,dtype=np.float32).reshape(-1)
    if factor==1:return a.copy()
    if not .9<=factor<=1.1:raise ValueError('Only small speech tempo changes are supported')
    if not ffmpeg or not Path(ffmpeg).is_file():raise ValueError('Local FFmpeg required for speech-only tempo')
    p=subprocess.run([str(ffmpeg),'-nostdin','-hide_banner','-loglevel','error','-f','f32le','-ar',str(sr),'-ac','1','-i','pipe:0','-af','atempo='+str(factor),'-f','f32le','pipe:1'],input=a.tobytes(),capture_output=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if p.returncode:raise ValueError('Speech tempo processing failed')
    b=np.frombuffer(p.stdout,dtype=np.float32).copy()
    if not len(b) or not np.isfinite(b).all():raise ValueError('Invalid speech tempo output')
    return b
