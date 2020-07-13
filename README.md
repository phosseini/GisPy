# Gist Implementation in Python
This repository is dedicated to the implementation of gist in Python. 

Based on Fuzzy-trace theory (FTT), when individuals read something, there are two mental representations encoded in parallel in their mind, including gist and verbatim. While verbatim is related to surface-level information in text, gist represents the bottom-meaning and underlying semantics of the text. Inspired by the Gist Inference Score (GIS) introduced by authors of FTT, we try to use text analysis and natural language processing tools to implement and improve GIS in Python. 

Our end goal is to have a metric that gives us an understanding of the potential and capacity of a piece of text in creating a coherent gist representation in people's when they read that text.

`GIS = Referential Cohesion + Deep Cohesion + (LSA Verb Overlap - WordNet Verb Overlap) - Word Concreteness - Imageability - Hypernymy Nouns & Verbs`

* **Index name:** **LSA verb Overlap (SMCAUSlsa)** and **WordNet Verb Overlap (SMCAUSwn)**
* **Status:** partially implemented
* **Explanation:** "A central dimension of forming a coherent situation model is the extent to which actions, as represented by verbs, are related to one another across a text. FTT suggests that abstract, rather than concrete verb overlap might help active readers construct gist situation models. Coh-Metrix uses **SMCAUSlsa** and **SMCAUSwn** to assess the extent to which verbs (actions) are interconnected across a text."
* **Implementation:** *SMCAUSlsa:* in the LSA algorithm, the cosine of two LSA vectors corresponding to the given pair of verbs is used to represent the degree of overlap of the two verbs. *SMCAUSwn:* In the WordNet algorithm, the overlap is a binary representation: `1` when two verbs are in the same synonym set and `0` otherwise.

---

* **Index name:** **Hypernymy for Nouns and Verbs (WRDHYPnv)**
* **Status:** implemented
* **Explanation:** "One type of relation in WordNet lexicon is the hypernym relation. Hypernym count is defined as the number of levels in a conceptual taxonomic hierarchy that is above (superordinate to) a word. For example, table (as an object) has seven hypernym levels: seat -> furniture -> furnishings -> instrumentality -> artifact -> object -> entity. A word having many hypernym levels tends to be more abstract. A lower value reflects an overall use of less specific words, while a higher value reflects an overall use of more specific words."
* **Implementation:** we list all NOUNs and VERBs in a document. Then for each NOUN/VERB, we find all synsets in WordNet to which the NOUN/VERB belongs to. For each synset, using the `hypernym_paths` we find the path length of the synset to its root hypernym. Since each NOUN/VERB may belong to more than one synset even with the same Part of Speech (POS), we repeat the previous step for all the synsets to which a NOUN/VERB belongs to. Finally, we compute the mean/average of all the path length values.

**Technical details:**
* In input text to the pipeline, one assumption is that paragraphs are separated by `\n`.
* In input string, documents are separated by `\n\n`.
