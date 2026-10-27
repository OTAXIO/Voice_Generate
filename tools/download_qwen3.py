"""Explicit public model download; credentials disabled, no remote code execution."""
import os,json
from pathlib import Path
os.environ.update(HF_HUB_DISABLE_XET='1',HF_HUB_OFFLINE='0',HF_HUB_DOWNLOAD_TIMEOUT='90',HF_HUB_ETAG_TIMEOUT='30')
from huggingface_hub import snapshot_download,HfApi
ROOT=Path(__file__).resolve().parents[1]
repo='Qwen/Qwen3-TTS-12Hz-0.6B-Base'
info=HfApi(token=False).model_info(repo)
target=ROOT/'models/Qwen3-TTS-12Hz-0.6B-Base'
snapshot_download(repo_id=repo,revision=info.sha,local_dir=str(target),token=False,max_workers=2,allow_patterns=['*.json','*.safetensors','*.txt','*.tiktoken','*.model','LICENSE','README.md'])
(target/'download-provenance.json').write_text(json.dumps(dict(repo_id=repo,revision=info.sha,source='official Hugging Face',token_used=False),indent=2),encoding='utf-8')
print('MODEL_READY '+str(target),flush=True)
