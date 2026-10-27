# -*- coding: utf-8 -*-
"""Small native Tk desktop front end; inference runs in a child process."""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import argparse,json,queue,subprocess,threading,datetime,traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from tkinter.scrolledtext import ScrolledText
from common import python_executable,environment

REPO=Path(__file__).resolve().parents[1]
LOCAL=REPO/'.local';LOCAL.mkdir(exist_ok=True)
SETTINGS=LOCAL/'desktop_settings.json'

class VoiceWindow:
    def __init__(self,root,project='',profile=None):
        self.root=root;root.title('Voice Generate · 本地语音生成');root.geometry('850x815');root.minsize(760,750)
        self.events=queue.Queue();self.proc=None;self.running=False;self.cancel_requested=False;self.result=None;self.raw=None;self.job_dir=None
        settings=json.loads(SETTINGS.read_text(encoding='utf-8')) if SETTINGS.exists() else {}
        self.values={k:tk.StringVar(value=str(settings.get(k,default))) for k,default in [('project',project),('gpt',''),('sovits',''),('reference',''),('text_file',''),('output_dir',str(Path.home()/'Documents/Voice_Generate')),('speed','1.05'),('pause','0.18'),('cleanup','标准'),('language','中文'),('temperature','0.75')]}
        if project:self.values['project'].set(project)
        style=ttk.Style();style.theme_use('clam')
        style.configure('.',font=('Microsoft YaHei UI',10),background='#f4f6fa',foreground='#243047')
        style.configure('TFrame',background='#f4f6fa');style.configure('TLabelframe',background='#f4f6fa',bordercolor='#dce2ec')
        style.configure('TLabelframe.Label',font=('Microsoft YaHei UI',10,'bold'))
        style.configure('TEntry',fieldbackground='white',padding=5)
        style.configure('TButton',padding=(10,6));style.configure('Primary.TButton',background='#2863d8',foreground='white',font=('Microsoft YaHei UI',11,'bold'))
        style.map('Primary.TButton',background=[('active','#1f50b5'),('disabled','#9eacc5')])
        body=ttk.Frame(root,padding=(20,14));body.pack(fill='both',expand=True)
        header=ttk.Frame(body);header.pack(fill='x')
        ttk.Label(header,text='本地语音生成',font=('Microsoft YaHei UI',19,'bold')).pack(side='left')
        ttk.Button(header,text='导入配置',command=self.import_profile).pack(side='right')
        ttk.Label(body,text='选择本地文件 → 生成 → 保存 WAV，全程无需网页。',foreground='#65748b').pack(anchor='w',pady=(4,12))
        files=ttk.LabelFrame(body,text='模型与素材',padding=10);files.pack(fill='x')
        self.path_row(files,0,'引擎目录','project',directory=True)
        self.path_row(files,1,'GPT 模型','gpt',types=[('GPT 权重','*.ckpt')])
        self.path_row(files,2,'SoVITS 模型','sovits',types=[('SoVITS 生成器权重','*.pth')])
        self.path_row(files,3,'参考音频','reference',types=[('音频','*.wav *.flac *.mp3 *.m4a'),('所有文件','*.*')])
        ttk.Label(files,text='参考音频原文',foreground='#46556d').grid(row=4,column=0,sticky='nw',pady=6)
        self.reference_text=tk.Text(files,height=2,wrap='word',font=('Microsoft YaHei UI',10),relief='solid',borderwidth=1,background='white')
        self.reference_text.grid(row=4,column=1,columnspan=2,sticky='ew',padx=(8,0),pady=5)
        self.reference_text.insert('1.0',settings.get('reference_text',''))
        ttk.Label(files,text='3～10 秒的完整台词；语气主要由参考音频决定。',foreground='#6f7c91').grid(row=5,column=1,columnspan=2,sticky='w',padx=8)
        text_box=ttk.LabelFrame(body,text='输入与输出',padding=10);text_box.pack(fill='x',pady=(10,0))
        self.path_row(text_box,0,'TXT 文件','text_file',types=[('文本文件','*.txt'),('所有文件','*.*')],after=self.preview_txt)
        self.path_row(text_box,1,'输出文件夹','output_dir',directory=True)
        self.preview=ScrolledText(text_box,height=3,wrap='word',font=('Microsoft YaHei UI',10),relief='solid',borderwidth=1,state='disabled',background='#fafbfd',foreground='#64718a')
        self.preview.grid(row=2,column=0,columnspan=3,sticky='ew',pady=(6,0))
        controls=ttk.Frame(body);controls.pack(fill='x',pady=(12,4))
        ttk.Label(controls,text='语速').pack(side='left');ttk.Spinbox(controls,from_=.8,to=1.3,increment=.05,width=5,textvariable=self.values['speed']).pack(side='left',padx=(6,15))
        ttk.Label(controls,text='句间停顿 / 秒').pack(side='left');ttk.Spinbox(controls,from_=0,to=1,increment=.05,width=5,textvariable=self.values['pause']).pack(side='left',padx=(6,15))
        ttk.Label(controls,text='句尾处理').pack(side='left');ttk.Combobox(controls,width=7,state='readonly',values=['关闭','轻度','标准','较强'],textvariable=self.values['cleanup']).pack(side='left',padx=(6,15))
        ttk.Combobox(controls,width=7,state='readonly',values=['中文','中英混合'],textvariable=self.values['language']).pack(side='left')
        ttk.Label(body,text='句尾处理会削减低能量尾音；若末字不完整，改为轻度或关闭。原始音频会一并保留。',foreground='#6f7c91',font=('Microsoft YaHei UI',9)).pack(anchor='w',pady=(0,10))
        actions=ttk.Frame(body);actions.pack(fill='x')
        self.generate_button=ttk.Button(actions,text='生成并保存',style='Primary.TButton',command=self.generate);self.generate_button.pack(side='left')
        self.stop_button=ttk.Button(actions,text='停止',command=self.cancel,state='disabled');self.stop_button.pack(side='left',padx=8)
        self.play_button=ttk.Button(actions,text='播放成品',command=lambda:self.play(self.result),state='disabled');self.play_button.pack(side='left')
        self.raw_button=ttk.Button(actions,text='播放原始',command=lambda:self.play(self.raw),state='disabled');self.raw_button.pack(side='left',padx=8)
        ttk.Button(actions,text='打开输出目录',command=self.open_output).pack(side='right')
        self.progress=ttk.Progressbar(body,mode='determinate');self.progress.pack(fill='x',pady=(12,5))
        self.status=tk.StringVar(value='就绪 · 使用本地模型，不启动网页服务')
        ttk.Label(body,textvariable=self.status,foreground='#2863b5',wraplength=790).pack(anchor='w')
        self.log=ScrolledText(body,height=4,font=('Microsoft YaHei UI',9),state='disabled',relief='flat',background='#eaf0f8',foreground='#53637b');self.log.pack(fill='both',expand=True,pady=(8,0))
        if profile:self.load_profile(profile)
        if self.values['text_file'].get():self.preview_txt()
        root.after(120,self.poll);root.protocol('WM_DELETE_WINDOW',self.close)

    def path_row(self,parent,row,label,key,directory=False,types=None,after=None):
        parent.columnconfigure(1,weight=1)
        ttk.Label(parent,text=label,width=12).grid(row=row,column=0,sticky='w',pady=4)
        ttk.Entry(parent,textvariable=self.values[key]).grid(row=row,column=1,sticky='ew',padx=8,pady=4)
        def browse():
            path=filedialog.askdirectory(parent=self.root) if directory else filedialog.askopenfilename(parent=self.root,filetypes=types)
            if path:self.values[key].set(path)
            if path and after:after()
        ttk.Button(parent,text='浏览…',width=7,command=browse).grid(row=row,column=2,pady=4)

    def import_profile(self):
        path=filedialog.askopenfilename(parent=self.root,filetypes=[('配置文件','*.json')])
        if path:
            try:self.load_profile(path)
            except Exception as error:messagebox.showerror('配置无法读取',str(error),parent=self.root)

    def load_profile(self,path):
        data=json.loads(Path(path).read_text(encoding='utf-8'));model=data['model'];request=data['request']
        for key,source in [('gpt','t2s_weights_path'),('sovits','vits_weights_path')]:self.values[key].set(model[source])
        self.values['reference'].set(request['ref_audio_path']);self.reference_text.delete('1.0','end');self.reference_text.insert('1.0',request.get('prompt_text',''))
        for key,source in [('speed','speed_factor'),('pause','fragment_interval'),('temperature','temperature')]:
            if source in request:self.values[key].set(str(request[source]))
        self.add_log('已导入模型与参考音频配置。')

    def preview_txt(self):
        try:
            path=Path(self.values['text_file'].get());data=path.read_bytes()
            text=data.decode('utf-16') if data.startswith((b'\xff\xfe',b'\xfe\xff')) else None
            if text is None:
                try:text=data.decode('utf-8-sig')
                except UnicodeDecodeError:text=data.decode('gb18030')
            self.preview.configure(state='normal');self.preview.delete('1.0','end');self.preview.insert('1.0',text[:3000]);self.preview.configure(state='disabled')
        except Exception as error:self.add_log('TXT 预览失败：'+str(error))

    def add_log(self,text):
        self.log.configure(state='normal');self.log.insert('end',text+'\n');self.log.see('end');self.log.configure(state='disabled')

    def save_settings(self):
        data={k:v.get() for k,v in self.values.items()};data['reference_text']=self.reference_text.get('1.0','end').strip()
        SETTINGS.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

    def generate(self):
        if self.running:return
        try:
            v={k:x.get().strip() for k,x in self.values.items()}
            for key in ['project','gpt','sovits','reference','text_file']:
                if not v[key] or not Path(v[key]).exists():raise ValueError('请选择有效的'+{'project':'引擎目录','gpt':'GPT 模型','sovits':'SoVITS 模型','reference':'参考音频','text_file':'TXT 文件'}[key])
            if not (Path(v['project'])/'GPT_SoVITS/TTS_infer_pack/TTS.py').is_file():raise ValueError('引擎目录中找不到 GPT_SoVITS/TTS_infer_pack/TTS.py。')
            reference_text=self.reference_text.get('1.0','end').strip()
            if not reference_text:raise ValueError('请填写与参考音频一致的原文。')
            speed=float(v['speed']);pause=float(v['pause'])
            if not .8<=speed<=1.3 or not 0<=pause<=1:raise ValueError('语速范围为 0.8～1.3，句间停顿为 0～1 秒。')
            if not v['output_dir']:raise ValueError('请选择输出文件夹。')
            output=Path(v['output_dir']).resolve();output.mkdir(parents=True,exist_ok=True)
            stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.job_dir=output/'.voice_jobs'/stamp;self.job_dir.mkdir(parents=True)
            job=dict(project=str(Path(v['project']).resolve()),gpt=str(Path(v['gpt']).resolve()),sovits=str(Path(v['sovits']).resolve()),reference=str(Path(v['reference']).resolve()),reference_text=reference_text,text_file=str(Path(v['text_file']).resolve()),output_file=str(output/(Path(v['text_file']).stem+'_'+stamp+'.wav')),speed=speed,pause=pause,cleanup={'关闭':'off','轻度':'light','标准':'standard','较强':'strong'}[v['cleanup']],language='zh' if v['language']=='中英混合' else 'all_zh',temperature=.75,seed=42)
            path=self.job_dir/'job.json';path.write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8');self.save_settings()
            self.running=True;self.cancel_requested=False;self.result=None;self.raw=None;self.progress.configure(value=0,maximum=1)
            self.generate_button.configure(state='disabled');self.stop_button.configure(state='normal');self.play_button.configure(state='disabled');self.raw_button.configure(state='disabled')
            self.status.set('正在启动本地生成……');self.add_log('开始生成：'+Path(v['text_file']).name)
            threading.Thread(target=self.run_worker,args=(path,Path(v['project'])),daemon=True).start()
        except Exception as error:messagebox.showerror('无法开始生成',str(error),parent=self.root)

    def run_worker(self,path,project):
        try:
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            self.proc=subprocess.Popen([python_executable(project),'-u','-B',str(REPO/'tools/generate_job.py'),'--job',str(path)],cwd=str(project),env=environment(project,self.job_dir),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=flags)
            if self.cancel_requested:
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(self.proc.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
                else:self.proc.terminate()
            with (self.job_dir/'engine.log').open('w',encoding='utf-8') as log:
                for line in self.proc.stdout:
                    log.write(line);log.flush()
                    if line.startswith('@@VOICE_EVENT '):self.events.put(json.loads(line[len('@@VOICE_EVENT '):]))
            self.events.put(dict(type='exit',code=self.proc.wait()))
        except Exception as error:self.events.put(dict(type='error',message=str(error)));self.events.put(dict(type='exit',code=1))

    def poll(self):
        try:
            while True:
                event=self.events.get_nowait();kind=event['type']
                if kind=='status':
                    self.status.set(event['message']);self.progress.configure(maximum=event['total'],value=event['current']);self.add_log(event['message'])
                elif kind=='done':
                    self.result=event['output'];self.raw=event['raw'];self.status.set(f"已保存 · {event['seconds']:.1f} 秒 · 句尾共整理 {event['removed_seconds']:.2f} 秒")
                    self.add_log('成品：'+self.result);self.play_button.configure(state='normal');self.raw_button.configure(state='normal')
                elif kind=='error':self.status.set('生成失败：'+event['message']);self.add_log('错误：'+event['message'])
                elif kind=='exit':
                    self.running=False;self.proc=None;self.generate_button.configure(state='normal');self.stop_button.configure(state='disabled')
                    if event['code'] and not self.result:self.add_log('生成已停止或失败；详细日志：'+str(self.job_dir/'engine.log'))
                (LOCAL/'gui_state.json').write_text(json.dumps(dict(running=self.running,event=event,result=self.result,raw=self.raw),ensure_ascii=False,indent=2),encoding='utf-8')
        except queue.Empty:pass
        self.root.after(120,self.poll)

    def cancel(self):
        self.cancel_requested=True
        if self.proc and self.proc.poll() is None:
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(self.proc.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            else:self.proc.terminate()
            self.status.set('已停止生成。');self.add_log('用户停止了生成。')

    def play(self,path):
        if path and Path(path).is_file():
            import winsound
            winsound.PlaySound(path,winsound.SND_FILENAME|winsound.SND_ASYNC)

    def open_output(self):
        folder=Path(self.values['output_dir'].get());folder.mkdir(parents=True,exist_ok=True);os.startfile(str(folder))

    def close(self):
        self.cancel();self.save_settings();self.root.destroy()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',default='');parser.add_argument('--profile');args=parser.parse_args()
    root=tk.Tk();VoiceWindow(root,args.project,args.profile);root.mainloop()

if __name__=='__main__':
    try:main()
    except Exception:
        (LOCAL/'desktop_error.log').write_text(traceback.format_exc(),encoding='utf-8');raise
