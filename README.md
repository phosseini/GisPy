# Gist Implementation in Python
This repository is dedicated to the implementation of gist in Python. 

Based on Fuzzy-trace theory (FTT), when individuals read something, there are two mental representations encoded in parallel in their mind, including gist and verbatim. While verbatim is related to surface-level information in text, gist represents the bottom-meaning and underlying semantics of the text. Inspired by the Gist Inference Score (GIS) introduced by authors of FTT, we try to use text analysis and natural language processing tools to implement and improve GIS in Python. 

Our end goal is to have a metric that gives us an understanding of the potential and capacity of a piece of text in creating a coherent gist representation in people's when they read that text.

`GIS = Referential Cohesion + Deep Cohesion + (LSA Verb Overlap - WordNet Verb Overlap) - Word Concreteness - Imageability - Hypernymy Nouns & Verbs`

* **Index name:** **LSA verb Overlap (SMCAUSlsa)** and **WordNet Verb Overlap (SMCAUSwn)**
* **Status:** under implementation
* **Explanation:** "A central dimension of forming a coherent situation model is the extent to which actions, as represented by verbs, are related to one another across a text. FTT suggests that abstract, rather than concrete verb overlap might help active readers construct gist situation models. Coh-Metrix uses **SMCAUSlsa** and **SMCAUSwn** to assess the extent to which verbs (actions) are interconnected across a text."
* **Implementation:** *SMCAUSlsa:* in the LSA algorithm, the cosine of two LSA vectors corresponding to the given pair of verbs is used to represent the degree of overlap of the two verbs. *SMCAUSwn:* In the WordNet algorithm, the overlap is a binary representation: `1` when two verbs are in the same synonym set and `0` otherwise.


**Technical details:**
* In input text to the pipeline, one assumption is that paragraphs are separated by `\n`.
* In input string, documents are separated by '\n\n'.
