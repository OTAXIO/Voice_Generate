"""Select the installed GPT-SoVITS interpreter and run a workflow command."""
import argparse,subprocess,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train','compare','synthesize','set_default'])
    parser.add_argument('--project',required=True)
    args,rest=parser.parse_known_args()
    project=Path(args.project).resolve()
    root=Path(__file__).resolve().parent
    python=project/'runtime/python.exe'
    executable=str(python) if python.is_file() else sys.executable
    # Resolve file arguments before switching to the engine working directory.
    paths={'--output','--labels','--features','--reference','--profile','--text-file','--gpt','--sovits','--spec'}
    for i,item in enumerate(rest[:-1]):
        if item in paths:rest[i+1]=str(Path(rest[i+1]).resolve())
    return subprocess.call([executable,'-u','-B',str(root/'tools'/(args.command+'.py')),'--project',str(project),*rest],cwd=str(project))

if __name__=='__main__':raise SystemExit(main())
