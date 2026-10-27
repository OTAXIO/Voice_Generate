import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from audio_cleanup import split_sentences,clean_phrase_end,read_local_text

class CleanupTests(unittest.TestCase):
    def test_sentence_boundaries_preserve_decimal_and_commas(self):
        self.assertEqual(split_sentences('先走十二点五公里，再检查。完成了吗？\n继续！'),['先走十二点五公里，再检查。','完成了吗？','继续！'])
        self.assertEqual(split_sentences('Value 3.14, keep going.\nNext line'),['Value 3.14, keep going.','Next line'])

    def test_breathy_tail_is_reduced_after_voiced_vowel(self):
        sr=16000;t=np.arange(sr)/sr;voice=.3*np.sin(2*np.pi*170*t)
        noise=np.random.RandomState(1).normal(0,.009,int(.42*sr))
        audio=np.r_[voice,noise,np.zeros(int(.1*sr))].astype(np.float32)
        clean,report=clean_phrase_end(audio,sr,'standard')
        self.assertTrue(report['breath_trimmed'])
        self.assertGreater(len(clean),int(1.05*sr))
        self.assertLess(len(clean),int(1.25*sr))
        np.testing.assert_array_equal(clean[:int(.9*sr)],audio[:int(.9*sr)])

    def test_voiced_ending_is_preserved(self):
        sr=16000;t=np.arange(sr)/sr;audio=(.2*np.sin(2*np.pi*170*t)).astype(np.float32)
        clean,report=clean_phrase_end(audio,sr,'standard')
        self.assertEqual(len(clean),len(audio));self.assertFalse(report['breath_trimmed'])

    def test_loud_unvoiced_release_is_not_clipped(self):
        sr=16000;t=np.arange(sr)/sr;voice=.2*np.sin(2*np.pi*170*t)
        release=np.random.RandomState(4).normal(0,.14,int(.3*sr))
        audio=np.r_[voice,release].astype(np.float32)
        clean,report=clean_phrase_end(audio,sr,'standard')
        self.assertFalse(report['breath_trimmed']);self.assertGreater(len(clean),int(1.25*sr))

    def test_off_is_exact_copy(self):
        audio=np.arange(100,dtype=np.float32)
        clean,report=clean_phrase_end(audio,16000,'off');np.testing.assert_array_equal(audio,clean)

if __name__=='__main__':unittest.main()
