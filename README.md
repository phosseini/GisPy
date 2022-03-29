# GisPy: Gist Inference Score Implementation in Python

**What is Gist?** Based on Fuzzy-trace theory (FTT), when individuals read a piece of text, there are two mental representations encoded in parallel in their mind including 1) `gist` and 2) `verbatim`. While verbatim is related to surface-level information in the text, gist represents the bottom-line meaning and underlying semantics of it. Inspired by the Gist Inference Score (GIS) formula introduced by [Wolfe et al. (2019)](https://link.springer.com/content/pdf/10.3758/s13428-019-01284-4.pdf) and implementation of coherence/cohesion indices by [Coh-Metrix](http://cohmetrix.com/), we leverage text analysis and Natural Language Processing (NLP) tools to implement and improve GIS in Python. 

Our goal is to develop a tool and metric that estimates the capacity of text in creating a coherent gist and mental representation in the human brain/mind.

### How to run GisPy
* Put all text documents separately as `.txt` files (one document per file) in the `/data/documents` folder.
   * Paragraphs in each document need to be spearated by [at least] one new line character (`\n`).  
* Now, run [`/gispy/run.py`](https://github.com/phosseini/gispy/blob/master/gispy/run.py) class: `python run.py [OUTPUT_FILE_NAME]`
    * `OUTPUT_FILE_NAME`: name of the output file in `.csv` format where results will be saved.

### Gist Inference Score (GIS) formula

```
GIS = Referential Cohesion 
      + Deep Cohesion 
      + (LSA Verb Overlap - WordNet Verb Overlap) 
      - Word Concreteness 
      - Imageability 
      - Hypernymy Nouns & Verbs
```
