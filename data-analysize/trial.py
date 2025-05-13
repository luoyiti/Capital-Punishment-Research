import spacy
nlp = spacy.load("en_core_web_lg")
trial_text = "I would like to apologize to the Nix family for taking Mark away from you. I hope this brings you closure. I want to say to all my family and friends around the world 'thank you' for supporting me. To my kids, stand tall and continue to make me proud. Don¡¯t worry about me, I'm ready to fly. Alright Warden, I¡¯m ready to ride."
doc = nlp(trial_text)
# 可视化依存关系
from spacy import displacy
sentence_spans = list(doc.sents)
# displacy.serve(sentence_spans, style="dep", auto_select_port=True)
displacy.serve(doc, style="dep", auto_select_port=True)
