"""Conservative phrase-end cleanup. Keeps a raw copy for listening comparisons."""
import re
from pathlib import Path
import numpy as np

def read_local_text(path):
    data=Path(path).read_bytes()
    if data.startswith((b'\xff\xfe',b'\xfe\xff')):return data.decode('utf-16')
    for encoding in ['utf-8-sig','gb18030']:
        try:return data.decode(encoding)
        except UnicodeDecodeError:pass
    raise ValueError('TXT encoding is unsupported; save as UTF-8.')

def split_sentences(text):
    # Full-stop/!? boundaries keep comma clauses together; decimal points stay intact.
    text=text.replace('\r\n','\n').replace('\r','\n')
    return [p.strip() for p in re.split(r'(?<=[。！？!?])\s*|(?<=\.)\s+|\n+',text) if p.strip()]

def clean_phrase_end(audio,sr,mode='standard'):
    original=np.asarray(audio,dtype=np.float32).reshape(-1)
    report={'mode':mode,'removed_seconds':0.0,'breath_trimmed':False}
    if mode=='off' or len(original)<int(.35*sr):return original.copy(),report
    settings={'light':(.35,.15,20.),'standard':(.65,.12,12.),'strong':(.9,.10,9.)}
    if mode not in settings:raise ValueError('Unknown tail cleanup mode')
    max_trim,keep,drop=settings[mode]
    hop=max(1,int(sr*.01));frame=max(hop,int(sr*.03))
    starts=np.arange(0,max(1,len(original)-frame+1),hop)
    rms=np.array([np.sqrt(np.mean(original[i:i+frame]**2)) for i in starts])
    peak=float(np.max(rms))
    if peak<1e-6:return original.copy(),report
    # First remove only near silence, retaining a short natural release.
    active=np.flatnonzero(rms>max(10**(-48/20),peak*10**(-36/20)))
    if not len(active):return original.copy(),report
    endpoint=min(len(original),int(starts[active[-1]]+frame+.06*sr))
    search_start=max(0,endpoint-int(1.3*sr))
    last_voiced=None
    for start,level in zip(starts,rms):
        if start<search_start or start>=endpoint:continue
        if level<max(10**(-40/20),peak*10**(-24/20)):continue
        # Autocorrelation is used as a voicing hint, never as speech recognition.
        x=original[start:start+frame];x=x-np.mean(x)
        step=max(1,int(sr/16000));x=x[::step];rate=sr/step
        if len(x)<32:continue
        fft=np.fft.rfft(x,n=2**int(np.ceil(np.log2(len(x)*2))))
        ac=np.fft.irfft(fft*np.conj(fft))[:len(x)]
        lo=max(1,int(rate/450));hi=min(len(ac)-1,int(rate/70))
        periodic=float(np.max(ac[lo:hi])/max(ac[0],1e-12)) if hi>lo else 0
        if periodic>=.38:last_voiced=int(start+frame)
    if last_voiced is not None:
        candidate=min(endpoint,last_voiced+int(keep*sr))
        tail=original[candidate:endpoint]
        tail_rms=float(np.sqrt(np.mean(tail**2))) if len(tail) else peak
        tail_duration=(endpoint-candidate)/sr
        # Do not cut loud releases, internal pauses, or arbitrarily long material.
        if .16<=tail_duration<=max_trim and tail_rms<peak*10**(-drop/20):
            endpoint=candidate;report['breath_trimmed']=True
    endpoint=max(int(.25*sr),endpoint)
    result=original[:endpoint].copy()
    fade=min(len(result),int(.008*sr))
    if fade:result[-fade:]*=np.linspace(1,0,fade,dtype=np.float32)
    report['removed_seconds']=(len(original)-len(result))/sr
    return result,report
