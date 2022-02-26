import spacy

texts = [
    "I am reading a book.",
    "I went home.",
]

nlp = spacy.load("en_core_web_trf")

for doc in nlp.pipe(texts):
    tokvecs = doc._.trf_data
    print(tokvecs.align)

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a sentence.")
displacy.serve(doc, style="dep")
