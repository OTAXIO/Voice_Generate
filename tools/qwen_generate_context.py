"""Continuous news-level synthesis followed by local sentence alignment."""
import argparse,json,os,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from prosody import synthesis_units
from speech_text import prepare_text

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    job_path=Path(a.job).resolve();job=json.loads(job_path.read_text(encoding='utf-8-sig'))
    text=prepare_text(Path(job['text_file']).read_text(encoding='utf-8'),job.get('pronunciation_readings'));sentences=synthesis_units(text)
    counts=job.get('context_group_sentence_counts') or [len(sentences)]
    if sum(counts)!=len(sentences) or any(x<=0 for x in counts):raise ValueError('Context grouping must cover every sentence exactly')
    run=job_path.parent/'context-source';run.mkdir(exist_ok=True)
    grouped=[];cursor=0
    for count in counts:grouped.append(''.join(sentences[cursor:cursor+count]));cursor+=count
    spoken=run/'spoken.txt';spoken.write_text('\n\n'.join(grouped),encoding='utf-8')
    aligned_text=run/'normalized-sentences.txt';aligned_text.write_text('\n'.join(sentences),encoding='utf-8')
    source=run/'context.wav';raw_job=dict(job,text_file=str(spoken),output_file=str(source),synthesis_unit='paragraph',max_context_chars=220,speech_rate=1.0,pause=0,max_new_tokens=int(job.get('context_max_new_tokens',1024)))
    source_job=run/'job.json';source_job.write_text(json.dumps(raw_job,ensure_ascii=False,indent=2),encoding='utf-8')
    env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NUMBA_CACHE_DIR=str(run/'numba'))
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    subprocess.run([sys.executable,'-u','-B',str(Path(__file__).with_name('qwen_generate_job.py')),'--job',str(source_job)],env=env,check=True,creationflags=flags,stdout=sys.stdout,stderr=sys.stderr)
    align_python=Path(job['alignment_python']);align_model=Path(job['alignment_model'])
    if not align_python.is_file() or not (align_model/'model.pt').is_file():raise ValueError('Local alignment runtime/model missing')
    aligned_job=dict(job,pause=float(job.get('pause',.4)),paragraph_pause=float(job.get('paragraph_pause',.6)),scene_break_after=[sum(counts[:i+1])-1 for i in range(len(counts)-1)])
    aligned_path=run/'aligned-job.json';aligned_path.write_text(json.dumps(aligned_job,ensure_ascii=False,indent=2),encoding='utf-8')
    command=[str(align_python),'-u','-B',str(Path(__file__).with_name('context_alignment.py')),'--source',str(source),'--source-meta',str(source.with_suffix('.json')),'--text',str(aligned_text),'--job',str(aligned_path),'--output',job['output_file'],'--fa-model',str(align_model)]
    subprocess.run(command,env=env,check=True,creationflags=flags,stdout=sys.stdout,stderr=sys.stderr)
    print('@@VOICE_EVENT '+json.dumps(dict(type='done',output=job['output_file'],backend='qwen3-tts-context'),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
