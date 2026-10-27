"""Align continuous Qwen speech, adjusting only verified quiet sentence gaps."""
import argparse,json,re,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from prosody import speech_tempo
from audio_cleanup import split_sentences
from speech_text import compound_spans

def alignment_tokens(text):
    import cn2an
    text=cn2an.transform(text.replace('ChatGPT','chat GPT'),'an2cn')
    return re.findall(r'[A-Za-z]+|[\u4e00-\u9fff]|[0-9]+',text)

def quiet_runs(audio,sr):
    hop=max(1,round(sr*.01));levels=np.array([np.sqrt(np.mean(audio[i:i+hop]**2)) for i in range(0,len(audio)-hop+1,hop)])
    mask=levels<.004;change=np.diff(np.r_[False,mask,False].astype(int))
    return [(int(a*hop),int(b*hop)) for a,b in zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)) if b-a>=8]

def select_sentence_gaps(audio,sr,sentences,timestamps):
    if sum(len(alignment_tokens(s)) for s in sentences)!=len(timestamps):raise ValueError('Alignment token count mismatch')
    if any(b<a or a<0 or b>len(audio)/sr*1000+100 for a,b in timestamps):raise ValueError('Invalid alignment times')
    runs=quiet_runs(audio,sr);selected=[];pos=0;previous=0
    for sentence in sentences[:-1]:
        pos+=len(alignment_tokens(sentence));anchor=(timestamps[pos-1][1]+timestamps[pos][0])/2000
        choices=[]
        for start,end in runs:
            if start<=previous or start<.1*sr or end>len(audio)-.1*sr:continue
            distance=max(start/sr-anchor,anchor-end/sr,0)
            if distance<=.85:choices.append((distance,-(end-start),start,end))
        if not choices:raise ValueError('No reliable quiet sentence boundary; do not cut active speech')
        _,_,start,end=min(choices);selected.append((start,end));previous=end
    return selected

def compound_quiet_gaps(source,sr,text,timestamps,limit=.12):
    if not .06<=limit<=.2:raise ValueError('Compound pause cap must be .06.. .2 seconds')
    runs=quiet_runs(source,sr);found=set()
    for start,end in compound_spans(text):
        first=len(alignment_tokens(text[:start]));last=first+len(alignment_tokens(text[start:end]))
        for i in range(first,last-1):
            if i+1>=len(timestamps):continue
            left,right=timestamps[i][1]/1000,timestamps[i+1][0]/1000
            if right<=left:continue
            anchor=(left+right)/2
            for a,b in runs:
                # Only quiet intervals wholly between aligned words are safe.
                if a/sr>=left-.02 and b/sr<=right+.02 and abs((a+b)/2/sr-anchor)<.18 and (b-a)/sr>limit+.04:found.add((a,b))
    return sorted(found)

def tighten_compounds(segment,sr,intervals,cap):
    pieces=[];cursor=0;removed=0
    for a,b in intervals:
        if not 0<=cursor<=a<b<=len(segment):raise ValueError('Compound gap outside sentence')
        if np.max(np.abs(segment[a:b]))>.02:raise ValueError('Refuse to shorten active compound speech')
        keep=min(round(.02*sr),(b-a)//2);target=round(cap*sr);middle=max(0,target-2*keep)
        pieces.extend([segment[cursor:a+keep],np.zeros(middle,dtype=np.float32),segment[b-keep:b]])
        removed+=(b-a)-2*keep-middle;cursor=b
    pieces.append(segment[cursor:]);return np.concatenate(pieces),removed

def assemble(source,sr,sentences,gaps,job,compound_gaps=None):
    breaks=set(job.get('scene_break_after',[]));default=float(job.get('pause',.4));paragraph=float(job.get('paragraph_pause',.6))
    rate=float(job.get('speech_rate',1));cursor=0;parts=[];raw_parts=[];entries=[];edits=[]
    for i,sentence in enumerate(sentences):
        if i<len(gaps):
            start,end=gaps[i];keep=min(round(.04*sr),(end-start)//2)
            stop=start+keep;next_start=end-keep
            desired=paragraph if i in breaks else default
            added=max(0,round(desired*sr)-2*keep)
        else:stop=len(source);next_start=stop;added=0;desired=0
        segment=source[cursor:stop].copy()
        if not len(segment):raise ValueError('Empty aligned sentence')
        within=[(a-cursor,b-cursor) for a,b in (compound_gaps or []) if cursor<=a<b<=stop]
        tightened,removed=tighten_compounds(segment,sr,within,float(job.get('compound_pause_max_seconds',.12)))
        processed=speech_tempo(tightened,sr,rate,job.get('ffmpeg_executable'))
        parts.append(processed);raw_parts.append(segment)
        entries.append(dict(sentence=sentence,raw_samples=len(segment),processed_samples=len(processed),mode='context_aligned',removed_seconds=0,speech_rate=rate,pause_after_samples=added,source_start_sample=cursor,source_end_sample=stop,compound_quiet_removed_samples=removed))
        if i<len(gaps):
            gap=np.zeros(added,dtype=np.float32);parts.append(gap);raw_parts.append(gap.copy())
            edits.append(dict(after_sentence=i,source_quiet_start=gaps[i][0],source_quiet_end=gaps[i][1],retained_edge_samples=keep,target_quiet_seconds=desired,inserted_samples=added,removed_quiet_samples=next_start-stop))
        cursor=next_start
    return np.concatenate(parts),np.concatenate(raw_parts),entries,edits

def main():
    p=argparse.ArgumentParser()
    for key in ['source','text','job','output','fa-model']:p.add_argument('--'+key,required=True)
    p.add_argument('--asr-model');p.add_argument('--source-meta');a=p.parse_args()
    import soundfile as sf,torch
    from funasr import AutoModel
    torch.set_num_threads(4)
    job=json.loads(Path(a.job).read_text(encoding='utf-8-sig'));text=Path(a.text).read_text(encoding='utf-8');sentences=split_sentences(text)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    tokenfile=output.with_suffix('.alignment.txt');tokens=alignment_tokens(text);tokenfile.write_text('utterance '+' '.join(tokens),encoding='utf-8')
    fa=AutoModel(model=a.fa_model,device='cpu',disable_update=True,disable_pbar=True)
    source,sr=sf.read(a.source,dtype='float32')
    timestamps=[];group_audit=[]
    if a.source_meta:
        source_meta=json.loads(Path(a.source_meta).read_text(encoding='utf-8'));cursor=0
        if source_meta['sample_rate']!=sr or source_meta['job'].get('pause',0)!=0:raise ValueError('Context source must have no inserted pauses')
        groups=source_meta['sentences']
    else:groups=[dict(sentence=text,processed_samples=len(source))];cursor=0
    for index,group in enumerate(groups):
        length=group['processed_samples'];group_audio=source[cursor:cursor+length]
        if len(group_audio)!=length:raise ValueError('Context source length mismatch')
        group_file=output.with_name(output.stem+'.align-group-'+str(index)+'.wav');group_text=group_file.with_suffix('.txt')
        sf.write(group_file,group_audio,sr);group_text.write_text('utterance '+' '.join(alignment_tokens(group['sentence'])),encoding='utf-8')
        result=fa.generate(input=(str(group_file),str(group_text)),data_type=('sound','text'),disable_pbar=True)
        if not result or not result[0].get('timestamp'):raise ValueError('No alignment timestamps')
        if len(result[0]['timestamp'])!=len(alignment_tokens(group['sentence'])):raise ValueError('Group alignment token mismatch')
        offset=cursor/sr*1000;timestamps.extend([[x+offset,y+offset] for x,y in result[0]['timestamp']])
        group_audit.append(dict(group=index,start_sample=cursor,samples=length,tokens=len(result[0]['timestamp'])))
        cursor+=length
    if cursor!=len(source):raise ValueError('Unmapped context source samples')
    gaps=select_sentence_gaps(source,sr,sentences,timestamps)
    compounds=compound_quiet_gaps(source,sr,text,timestamps,float(job.get('compound_pause_max_seconds',.12)))
    audio,raw,entries,edits=assemble(source,sr,sentences,gaps,job,compounds)
    sf.write(output,audio,sr);rawpath=output.with_name(output.stem+'_原始.wav');sf.write(rawpath,raw,sr)
    audit=dict(model=a.fa_model,method='local text alignment refined to quiet waveform intervals',timestamps=timestamps,groups=group_audit,sentence_gaps=edits,compound_quiet_gaps=compounds,compound_pause_max_seconds=job.get('compound_pause_max_seconds',.12))
    if a.asr_model:
        asr=AutoModel(model=a.asr_model,device='cpu',disable_update=True,disable_pbar=True)
        audit['asr_result']=asr.generate(input=a.source,disable_pbar=True)
    output.with_suffix('.alignment.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    import hashlib
    meta=dict(backend='qwen3-tts-context',model_path=job['model_path'],job=job,sentences=entries,sample_rate=sr,seconds=len(audio)/sr,raw_seconds=len(raw)/sr,raw_path=str(rawpath),output_path=str(output),input_sha256=hashlib.sha256(Path(a.text).read_bytes()).hexdigest(),voice_quality_approved=False,alignment_audit=str(output.with_suffix('.alignment.json')),generation_mode='continuous context, then quiet-boundary pause adjustment')
    output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),seconds=meta['seconds'],sentences=len(entries),pause=job.get('pause',.4),paragraph_pause=job.get('paragraph_pause',.6)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
