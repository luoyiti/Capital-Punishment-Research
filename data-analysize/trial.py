import spacy
nlp = spacy.load("en_core_web_lg")
trial_text = "I apologize to the Nix family for taking Mark away from you. I hope this brings you closure. I want to say to all my family and friends around the world 'thank you' for supporting me. To my kids, stand tall and continue to make me proud. Don¡¯t worry about me, I'm ready to fly. Alright Warden, I¡¯m ready to ride."
doc = nlp(trial_text)
# 可视化依存关系
from spacy import displacy
sentence_spans = list(doc.sents)
# displacy.serve(sentence_spans, style="dep", auto_select_port=True)
# ...existing code...
options = {
    "compact": True,
    "bg": "#f0f0f0",  # A light grey background
    "color": "#333333", # Darker text
    "font": "Arial",
    "distance": 150,
    "arrow_stroke": 1,
    "arrow_width": 8
}
displacy.serve(doc, style="dep", auto_select_port=True, options=options)
