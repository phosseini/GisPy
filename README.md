# Gist Implementation in Python
This repository is dedicated to the implementation of gist in Python. 

**What is Gist?** Based on Fuzzy-trace theory (FTT), when individuals read a piece of text, there are two mental representations encoded in parallel in their mind, including gist and verbatim. While verbatim is related to surface-level information in the text, gist represents the bottom-meaning and underlying semantics of it. Inspired by the Gist Inference Score (GIS), we leverage text analysis and Natural Language Processing (NLP) tools to implement and improve GIS in Python. 

Our end goal is to have a metric that gives us an understanding of the potential and capacity of a piece of text in creating a coherent gist representation in the human brain/mind.

`GIS = Referential Cohesion + Deep Cohesion + (LSA Verb Overlap - WordNet Verb Overlap) - Word Concreteness - Imageability - Hypernymy Nouns & Verbs`

| Index name | Status | Explanation | Implementation Detail |
| :----: | :----: | :---- | :---- |
| LSA verb Overlap (**SMCAUSlsa**), WordNet Verb Overlap (**SMCAUSwn**) | :white_check_mark: | "A central dimension of forming a coherent situation model is the extent to which actions, as represented by verbs, are related to one another across a text. FTT suggests that abstract, rather than concrete verb overlap might help active readers construct gist situation models. Coh-Metrix uses **SMCAUSlsa** and **SMCAUSwn** to assess the extent to which verbs (actions) are interconnected across a text." | *SMCAUSlsa:* in the LSA algorithm, the cosine of two LSA vectors corresponding to the given pair of verbs is used to represent the degree of overlap between two verbs. *SMCAUSwn:* In the WordNet algorithm, the overlap is a binary representation: `1` when two verbs are in the same synonym set and `0` otherwise. To improve the SMCAUSlsa, we implemented a new metric called **SMCAUSlme** that leverages word embedding from pre-trained language models (e.g., BERT) instead of LSA vectors to compute the overlap among verbs in a document. By default, SMCAUSlme computes the mean of cosine similarity among all pairs of VERBs in a document. However, it can also receive a list of Part of Speech (POS) tags instead of just VERBs and compute the similarity among all pairs of words with those POS tags. |
| Hypernymy for Nouns and Verbs (**WRDHYPnv**) | :white_check_mark: | "One type of relation in WordNet lexicon is the hypernym relation. Hypernym count is defined as the number of levels in a conceptual taxonomic hierarchy that is above (superordinate to) a word. For example, table (as an object) has seven hypernym levels: `seat -> furniture -> furnishings -> instrumentality -> artifact -> object -> entity`. A word having many hypernym levels tends to be more abstract. A lower value reflects an overall use of less specific words, while a higher value reflects an overall use of more specific words." | We list all NOUNs and VERBs in a document. Then for each NOUN/VERB, we find all synsets in WordNet to which the NOUN/VERB belongs. For each synset, using the `hypernym_paths` we find the path length of the synset to its root hypernym. Since each NOUN/VERB may belong to more than one synset even with the same Part of Speech (POS), we repeat the previous step for all the synsets to which a NOUN/VERB belongs. Finally, we compute the mean/average of all the path length values.|
| Concreteness for content words (**WRDCNCc**), Imagability for content words (**WRDIMGc**) | :white_check_mark: | -- | -- |
| Text Easability PC Referential cohesion (**PCREFz**) | :x: | "A text with high referential cohesion contains words and ideas that overlap across sentences and the entire text, forming explicit threads that connect the text for the reader. Low-cohesion text is typically more difficult to process because there are fewer connections that tie the ideas together for the reader." | -- |
| Text Easability PC Deep cohesion (**PCDCz**) | :x: | "This dimension reflects the degree to which the text contains causal and intentional connectives when there are causal and logical relationships within the text. These connectives help the reader form a deeper and more coherent understading of the causal events, processes, and actions in the text. When a text contains many relationships but does not contain those connectives, the reader must infer the relationships between the ideas in the text. If the text is high in deep cohesion, then those relationships and global cohesion are more explicit." | -- |

**Technical details:**
* In input text to the pipeline, one assumption is that paragraphs and documents are separated by `\n` and `\n\n`, respectively.
