"""Run training in an isolated overlay; original installation is read only."""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import argparse,json,subprocess,time,os
from pathlib import Path
from common import environment,install_overlay,python_executable

parser=argparse.ArgumentParser()
parser.add_argument('--project',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--stage',choices=['all','s1','s2'],default='all')
args=parser.parse_args()
project=Path(args.project).resolve();output=Path(args.output).resolve()
if (output/'STOP').exists():parser.error('STOP marker present; remove it explicitly before resuming.')
code=install_overlay(project,output);env=environment(project,output)
status=json.loads((output/'status.json').read_text(encoding='utf-8')) if (output/'status.json').exists() else {'data_ready':True,'s2_complete':False,'s1_complete':False}
def save(): (output/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
for stage,script,flag,config in [('s2','s2_train.py','--config','s2.json'),('s1','s1_train.py','--config_file','s1.yaml')]:
    if args.stage not in ['all',stage]:continue
    if (output/'STOP').exists():break
    if not (output/config).is_file():parser.error('Missing '+config+'; run prepare first.')
    with (output/(stage+'.log')).open('w',encoding='utf-8') as log:
        proc=subprocess.Popen([python_executable(project),'-u','-B',str(code/script),flag,str(output/config)],cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT)
        status.update(stage=stage,pid=proc.pid);save()
        try:
            while proc.poll() is None:
                if (output/'STOP').exists():raise KeyboardInterrupt
                time.sleep(2)
        except KeyboardInterrupt:
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
            else:proc.terminate()
            proc.wait();status['stopped']=True;save();raise SystemExit(130)
        status[stage+'_complete']=proc.returncode==0;save()
        if proc.returncode:raise SystemExit('Training failed: '+str(output/(stage+'.log')))
    print(stage+' complete',flush=True)
