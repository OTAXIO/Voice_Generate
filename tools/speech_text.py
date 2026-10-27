"""TTS-only lexical readings and joined model names; public spelling is untouched."""
import re

# Chinese homophones constrain a known word, not every occurrence of 调.
READINGS={'调用':'掉用'}
COMPOUND=re.compile(r'(?i)(?:GPT[- ]?[0-9]+(?:\.[0-9]+)*\s*(?:Terra|Astra|Sol)|[0-9]+(?:\.[0-9]+)+\s*(?:Terra|Astra|Sol)|ChatGPT\s+Work|Open\s+AI)')

def prepare_text(text,readings=None):
    rules=dict(READINGS);rules.update(readings or {})
    for word,value in sorted(rules.items(),key=lambda x:len(x[0]),reverse=True):text=text.replace(word,value)
    # An English space inside these names never creates a synthesis unit.
    def join(m):
        value=m.group()
        return re.sub(r'(?<=\d)\s+(?=(?:Terra|Astra|Sol)\b)','',value,flags=re.I)
    return COMPOUND.sub(join,text)

def compound_spans(text):
    return [m.span() for m in COMPOUND.finditer(text)]
