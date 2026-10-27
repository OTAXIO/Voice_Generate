import unittest,sys,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prosody import synthesis_units,sampling,speech_tempo
from prepare_qwen_dataset import export

class ProsodyChecks(unittest.TestCase):
    def test_small_tempo_change_preserves_pitch_and_pause_samples(self):
        import numpy as np
        ff=Path('D:/others/TOOLS/GPT-SoVITS-v2pro-20250604/runtime/ffmpeg.exe')
        if not ff.is_file():self.skipTest('Local FFmpeg not installed')
        sr=24000;t=np.arange(sr*2)/sr;audio=(.15*np.sin(2*np.pi*220*t)).astype(np.float32)
        changed=speech_tempo(audio,sr,1.03,ff)
        self.assertAlmostEqual(len(changed)/sr,2/1.03,delta=.04)
        frequency=np.argmax(np.abs(np.fft.rfft(changed)))*sr/len(changed)
        self.assertAlmostEqual(frequency,220,delta=2)
        pause=np.zeros(round(sr*.2),dtype=np.float32)
        combined=np.concatenate([changed,pause,changed])
        self.assertTrue(np.array_equal(combined[len(changed):len(changed)+len(pause)],pause))

    def test_context_keeps_sentences_and_blank_line_boundaries(self):
        text='第一句话。第二句话。\n\n第三段。'
        self.assertEqual(synthesis_units(text,'paragraph'),['第一句话。 第二句话。','第三段。'])
        self.assertEqual(synthesis_units(text),['第一句话。','第二句话。','第三段。'])
    def test_no_word_or_version_is_truncated(self):
        text='Open AI version 3.14 has arrived. Follow the account page.'
        units=synthesis_units(text,'paragraph',30)
        self.assertIn('3.14',units[0]);self.assertEqual(' '.join(units),text)
    def test_defaults_and_invalid_sampling(self):
        self.assertEqual(sampling({})['top_k'],5)
        with self.assertRaises(ValueError):sampling(dict(top_p=0))
    def test_dataset_keeps_holdout_and_detects_duplicate_audio(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);(r/'a.wav').write_bytes(b'a');(r/'b.wav').write_bytes(b'b');(r/'ref.wav').write_bytes(b'ref')
            rows=[dict(file='a.wav',text='甲',split='train'),dict(file='b.wav',text='乙',split='holdout')]
            (r/'manifest.json').write_text(json.dumps(rows),encoding='utf-8')
            report=export(r/'manifest.json',r,r/'ref.wav',r/'out')
            self.assertEqual(report['counts'],dict(train=1,validation=1));self.assertFalse(report['trained'])
            (r/'b.wav').write_bytes(b'a')
            with self.assertRaises(ValueError):export(r/'manifest.json',r,r/'ref.wav',r/'bad')
