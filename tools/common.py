import os, shutil, sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]

def install_overlay(project: Path, output: Path):
    code = output / 'code'
    code.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project/'GPT_SoVITS/AR',code/'AR',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for source in (REPOSITORY/'overrides/GPT_SoVITS').rglob('*.py'):
        dest=code/source.relative_to(REPOSITORY/'overrides/GPT_SoVITS')
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    return code

def environment(project: Path, output: Path):
    env=os.environ.copy()
    temp=output/'temp';temp.mkdir(parents=True,exist_ok=True)
    env.update(PYTHONIOENCODING='utf-8',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',version='v2Pro',hz='25hz',TEMP=str(temp),TMP=str(temp),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NUMBA_CACHE_DIR=str(temp/'numba'),MPLCONFIGDIR=str(temp/'mpl'))
    env['PATH']=str(project/'runtime')+os.pathsep+env.get('PATH','')
    env['PYTHONPATH']=os.pathsep.join(map(str,[output/'code',project,project/'GPT_SoVITS',project/'GPT_SoVITS/BigVGAN',project/'GPT_SoVITS/eres2net']))
    return env

def python_executable(project: Path):
    bundled=project/'runtime/python.exe'
    return str(bundled) if bundled.is_file() else sys.executable
