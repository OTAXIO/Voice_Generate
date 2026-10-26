import importlib.util,sys,unittest,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.test-artifacts';CACHE.mkdir(exist_ok=True)
os.environ.update(TEMP=str(CACHE),TMP=str(CACHE),NUMBA_CACHE_DIR=str(CACHE/'numba'),MPLCONFIGDIR=str(CACHE/'mpl'))
import torch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class RuntimeFixTests(unittest.TestCase):
    def test_learning_rate_warms_up_and_decays(self):
        module=load('voice_scheduler',ROOT/'overrides/GPT_SoVITS/AR/modules/lr_schedulers.py')
        param=torch.nn.Parameter(torch.zeros(1));optimizer=torch.optim.Adam([param],lr=.01)
        scheduler=module.WarmupCosineLRSchedule(optimizer,1e-5,1e-4,1e-5,warmup_steps=2,total_steps=6)
        values=[]
        for _ in range(7):scheduler.step();values.append(optimizer.param_groups[0]['lr'])
        self.assertAlmostEqual(values[0],1e-5)
        self.assertAlmostEqual(values[2],1e-4)
        self.assertAlmostEqual(values[-1],1e-5)
        self.assertTrue(values[0]<values[1]<values[2])
        self.assertTrue(values[2]>values[3]>values[4])
        self.assertEqual(scheduler.get_last_lr(),[values[-1]])

    def test_unicode_checkpoint_round_trip(self):
        module=load('voice_utils',ROOT/'overrides/GPT_SoVITS/utils.py')
        folder=ROOT/'.test-artifacts/中文目录';folder.mkdir(parents=True,exist_ok=True)
        target=folder/'管理员.pth';module.my_save({'weight':torch.arange(4)},str(target))
        result=torch.load(str(target),map_location='cpu',weights_only=False)
        self.assertTrue(torch.equal(result['weight'],torch.arange(4)))

    def test_gpu_sampler_without_process_group(self):
        from unittest.mock import patch
        module=load('voice_bucket_sampler',ROOT/'overrides/GPT_SoVITS/AR/data/bucket_sampler.py')
        class Dataset:
            def __len__(self):return 8
            def get_sample_length(self,index):return 2.0+index*.1
        with patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'set_device'),patch.object(torch.distributed,'is_initialized',return_value=False):
            sampler=module.DistributedBucketSampler(Dataset(),batch_size=2)
        self.assertEqual(sampler.num_replicas,1)
        self.assertEqual(sampler.rank,0)
        self.assertEqual(set(iter(sampler)),set(range(8)))

    def test_default_selection_updates_ui_config_and_cache(self):
        from unittest.mock import patch
        import yaml
        module=load('voice_default',ROOT/'tools/set_default.py')
        project=CACHE/'fake_engine';(project/'GPT_SoVITS/configs').mkdir(parents=True,exist_ok=True)
        (project/'weights').mkdir(exist_ok=True)
        for file in ['admin.ckpt','admin.pth']:(project/'weights'/file).write_bytes(b'test')
        original='GPT_dropdown = gr.Dropdown(\n    value=GPT_names[-1],\n)\nSoVITS_dropdown = gr.Dropdown(\n    value=SoVITS_names[0],\n)\n'
        (project/'webui.py').write_text(original,encoding='utf-8')
        (project/'GPT_SoVITS/configs/tts_infer.yaml').write_text('custom: {}\n',encoding='utf-8')
        (project/'weight.json').write_text('{"GPT": {}, "SoVITS": {"v2": "missing.pth"}}',encoding='utf-8')
        model={'version':'v2Pro','t2s_weights_path':str(project/'weights/admin.ckpt'),'vits_weights_path':str(project/'weights/admin.pth')}
        profile=CACHE/'test_profile.json';profile.write_text(json.dumps({'model':model}),encoding='utf-8')
        backup=CACHE/'ui_backup'
        if (backup/'webui.py').exists():(backup/'webui.py').unlink()
        with patch.object(sys,'argv',['set_default','--project',str(project),'--output',str(backup),'--profile',str(profile)]):module.main()
        self.assertEqual((backup/'webui.py').read_text(encoding='utf-8'),original)
        source=(project/'webui.py').read_text(encoding='utf-8')
        namespace={'GPT_names':['other.ckpt','weights/admin.ckpt'],'SoVITS_names':['other.pth','weights/admin.pth']}
        import ast
        for assignment in ast.parse(source).body:
            value=next(k.value for k in assignment.value.keywords if k.arg=='value')
            selected=eval(compile(ast.Expression(value),'default','eval'),namespace)
            self.assertIn(selected,['weights/admin.ckpt','weights/admin.pth'])
        settings=yaml.safe_load((project/'GPT_SoVITS/configs/tts_infer.yaml').read_text(encoding='utf-8'))
        self.assertEqual(settings['custom'],model)
        cache=json.loads((project/'weight.json').read_text(encoding='utf-8'))
        self.assertNotIn('v2',cache['SoVITS'])
        self.assertEqual(cache['SoVITS']['v2Pro'],'weights/admin.pth')

if __name__=='__main__':unittest.main()
