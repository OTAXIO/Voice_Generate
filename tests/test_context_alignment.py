import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from context_alignment import select_sentence_gaps,assemble

class ContextAlignmentChecks(unittest.TestCase):
    def fixture(self):
        sr=1000;a=(.2*np.sin(2*np.pi*50*np.arange(2000)/sr)).astype(np.float32);a[700:1300]=0
        return a,sr,['甲乙。','丙丁。'],[[0,350],[350,700],[1300,1600],[1600,1950]]
    def test_sentence_pause_changes_only_quiet_region(self):
        a,sr,text,ts=self.fixture();gaps=select_sentence_gaps(a,sr,text,ts)
        changed,raw,entries,audit=assemble(a,sr,text,gaps,dict(pause=.4,speech_rate=1))
        self.assertTrue(np.array_equal(changed[:700],a[:700]))
        self.assertTrue(np.array_equal(changed[1100:],a[1300:]))
        self.assertTrue(np.all(changed[700:1100]==0));self.assertEqual(len(changed),1800)
        self.assertEqual(sum(e['processed_samples']+e['pause_after_samples'] for e in entries),len(changed))
        self.assertEqual(audit[0]['target_quiet_seconds'],.4)
    def test_module_pause_can_be_longer(self):
        a,sr,text,ts=self.fixture();gaps=select_sentence_gaps(a,sr,text,ts)
        changed,_,_,_=assemble(a,sr,text,gaps,dict(pause=.4,paragraph_pause=.6,scene_break_after=[0]))
        self.assertEqual(len(changed),2000);self.assertTrue(np.all(changed[700:1300]==0))
    def test_no_silent_cut_when_alignment_has_no_quiet_boundary(self):
        a,sr,text,ts=self.fixture();a[700:1300]=.1
        with self.assertRaises(ValueError):select_sentence_gaps(a,sr,text,ts)
    def test_alignment_token_mismatch_stops(self):
        a,sr,text,ts=self.fixture()
        with self.assertRaises(ValueError):select_sentence_gaps(a,sr,text,ts[:-1])

class CompoundGapChecks(unittest.TestCase):
    def test_shortens_only_quiet_in_name_and_preserves_sentence_pause(self):
        from context_alignment import assemble
        sr=1000;a=np.ones(2500,dtype=np.float32)*.1;a[300:700]=0;a[1100:1700]=0
        audio,raw,entries,_=assemble(a,sr,['模型。','第二句。'],[(1100,1700)],dict(pause=.4,speech_rate=1,compound_pause_max_seconds=.12),[(300,700)])
        self.assertEqual(entries[0]['compound_quiet_removed_samples'],280);self.assertTrue(np.array_equal(audio[:300],a[:300]));self.assertEqual(entries[0]['pause_after_samples'],320);self.assertEqual(len(raw)-len(audio),280)
    def test_refuses_to_cut_voiced_audio(self):
        from context_alignment import tighten_compounds
        with self.assertRaises(ValueError):tighten_compounds(np.ones(1000,dtype=np.float32)*.1,1000,[(200,600)],.12)
