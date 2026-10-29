import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from speech_text import prepare_text,compound_spans
class SpeechTextChecks(unittest.TestCase):
    def test_operator_fourth_tone_only_in_tts(self):
        original='新增干员、六星干员和干员展示。干净、干活、干涉。'
        self.assertEqual(prepare_text(original),'新增赣员、六星赣员和赣员展示。干净、干活、干涉。')
        self.assertIn('干员展示',original)
        from context_alignment import alignment_tokens
        self.assertEqual(len(alignment_tokens(prepare_text('干员展示。'))),len(alignment_tokens('干员展示。')))
    def test_polyphonic_correction_is_lexical_and_tts_only(self):
        original='调用 API，调整参数。'
        self.assertEqual(prepare_text(original),'掉用 API，调整参数。');self.assertEqual(original,'调用 API，调整参数。')
    def test_model_name_whitespace_does_not_split_units(self):
        from prosody import synthesis_units
        text=prepare_text('GPT-6.1 Terra 已发布。现在调用模型。')
        self.assertIn('6.1Terra',text);self.assertEqual(len(synthesis_units(text)),2);self.assertEqual(prepare_text(text),text);self.assertTrue(compound_spans(text))
