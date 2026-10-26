"""Set an explicit voice in the original WebUI, keeping recoverable backups."""
import argparse,ast,json,re,shutil
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['project','output','profile']:parser.add_argument('--'+flag,required=True)
    args=parser.parse_args();project=Path(args.project).resolve();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    profile=json.loads(Path(args.profile).read_text(encoding='utf-8'));model=profile['model']
    names={}
    for key in ['t2s_weights_path','vits_weights_path']:
        p=Path(model[key]).resolve()
        if not p.is_file():raise FileNotFoundError(p)
        try:names[key]=p.relative_to(project).as_posix()
        except ValueError:raise ValueError('WebUI weights must be inside the engine installation; copy them into its weights folders first.')
    webui=project/'webui.py';source=webui.read_text(encoding='utf-8');lines=source.splitlines(keepends=True)
    assignments={n.targets[0].id:n.value for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['GPT_dropdown','SoVITS_dropdown']}
    for target,key,choices,fallback in [('GPT_dropdown','t2s_weights_path','GPT_names','-1'),('SoVITS_dropdown','vits_weights_path','SoVITS_names','0')]:
        if target not in assignments:raise ValueError('Unsupported WebUI source: missing '+target)
        node=next(k.value for k in assignments[target].keywords if k.arg=='value')
        if node.lineno!=node.end_lineno:raise ValueError('Unsupported multiline default expression')
        preferred=json.dumps(names[key],ensure_ascii=True);expression=f'{preferred} if {preferred} in {choices} else {choices}[{fallback}]'
        indent=re.match(r'\s*',lines[node.lineno-1]).group(0)
        lines[node.lineno-1]=indent+'value=('+expression+'),\n'
    updated=''.join(lines);compile(updated,str(webui),'exec')
    import yaml
    cfg=project/'GPT_SoVITS/configs/tts_infer.yaml';cache=project/'weight.json'
    for path,label in [(webui,'webui.py'),(cfg,'tts_infer.yaml'),(cache,'weight.json')]:
        backup=output/label
        if path.exists() and not backup.exists():shutil.copy2(path,backup)
    settings=yaml.safe_load(cfg.read_text(encoding='utf-8'));settings['custom']=model
    weights=json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else {'GPT':{},'SoVITS':{}}
    for category in ['GPT','SoVITS']:
        weights.setdefault(category,{})
        for version,path in list(weights[category].items()):
            if isinstance(path,str) and not (project/path).exists():del weights[category][version]
    weights['GPT'][model['version']]=names['t2s_weights_path'];weights['SoVITS'][model['version']]=names['vits_weights_path']
    webui.write_text(updated,encoding='utf-8');cfg.write_text(yaml.safe_dump(settings,allow_unicode=True),encoding='utf-8');cache.write_text(json.dumps(weights,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'applied.json').write_text(json.dumps(names,ensure_ascii=False,indent=2),encoding='utf-8')
    print('WebUI defaults and API config updated; backups:',output)

if __name__=='__main__':main()
