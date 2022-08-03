# GisPy: A Tool for Measuring Gist Inference Score in Text

<p align="center">
  <img src='gispy.png' width='450' height='400' style="vertical-align:middle;margin:100px 50px">
</p>

**What is Gist?** Based on Fuzzy-trace theory (FTT), when individuals read a piece of text, there are two mental representations encoded in parallel in their mind including 1) **gist** and 2) **verbatim**. While verbatim is related to surface-level information in the text, gist represents the bottom-line meaning and underlying semantics of it.

Inspired by the definition of Gist Inference Score (GIS) by [Wolfe et al. (2019)](https://link.springer.com/content/pdf/10.3758/s13428-019-01284-4.pdf) and implementation of coherence/cohesion indices in [Coh-Metrix](http://cohmetrix.com/), we developed `GisPy`, a tool for measuring GIS in text.

### How to run GisPy
1. Install the requirements: `pip install -r requirements.txt`
   * We suggest you create a new virtual environment (e.g., a [conda enviroment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)).
   * If you only want to run GisPy and don't need to run jupyter notebooks, you can skip installing the following packages:
      * `matplotlib, textract, wayback`
2. Install the spaCy model: `python -m spacy download en_core_web_trf`  
3. Put all text documents separately as `.txt` files (one document per file) in the `/data/documents` folder.
   * Paragraphs in each document need to be spearated by [at least] one new line character (`\n`).  
4. Run [`/gispy/run.py`](https://github.com/phosseini/gispy/blob/master/gispy/run.py) class: `python run.py [OUTPUT_FILE_NAME]`
    * `OUTPUT_FILE_NAME`: name of the output file in `.csv` format where results will be saved.
5. The output file contains the following information:
    * GIS score for each document in a column named `gis`
    * Indices and the z-scores of indices

### :information_source: Important
GIS will be computed based on the indices listed in [gis_config.json](https://github.com/phosseini/GisPy/blob/master/gispy/gis_config.json) file. This file is a dictionary of indices with their associated weights to give you maximum flexibility about how to use GisPy indices when computing the GIS scores. You can pick any of the indices from the following table. By default in the config file, we have listed the indices that are used in the original GIS formula. Format of the config file is like the following:
```
{
  "index_1": weight of index_1,
  ...
  "index_n": weight of index_n
}
```
An example:
```
{
  "PCREF_ap": 1,
  "PCDC": 1,
  "SMCAUSe_1p": 1,
  "SMCAUSwn_a_binary": -1,
  "PCCNC_megahr": -1,
  "WRDIMGc_megahr": -1,
  "WRDHYPnv": -1
}
```
`weight` is a real number that will be multiplied by the mean of index values when we linearly combine the index values in the GIS formula. If you want to ignore an index, you can either not include it in the dictionary at all, or you can simply set its `weight` to `0`.


### List of GisPy indices
In the following, there is a list of all indices generated by/in GisPy. To make it easier to map these indices with Coh-Metrix indices, we mainly followed Coh-Metrix indices’ names with some minor modifications (e.g., using different postfixes to show the exact implementation method for each index if there are multiple implementations).

| Index | Implementations |
| :---: | :---:|
| Number of Paragraphs | `DESPC` |
| Number of Sentences | `DESSC` |
| Referential Cohesion | `CoREF`, `PCREF_1`, `PCREF_a`, `PCREF_1p`, `PCREF_ap` |
| Deep Cohesion | `PCDC` |
| Semantic Verb Overlap | `SMCAUSe_1`, `SMCAUSe_a`, `SMCAUSe_1p`, `SMCAUSe_ap` |
| WordNet Verb Overlap | `SMCAUSwn_1p_path`, `SMCAUSwn_1p_lch`, `SMCAUSwn_1p_wup`, `SMCAUSwn_1p_binary`, `SMCAUSwn_ap_path`, `SMCAUSwn_ap_lch`, `SMCAUSwn_ap_wup`, `SMCAUSwn_ap_binary`, `SMCAUSwn_1_path`, `SMCAUSwn_1_lch`, `SMCAUSwn_1_wup`, `SMCAUSwn_1_binary`, `SMCAUSwn_a_path`, `SMCAUSwn_a_lch`, `SMCAUSwn_a_wup`, `SMCAUSwn_a_binary` |
| Word Concreteness | `PCCNC_megahr`, `PCCNC_mrc` |
| Imageability | `WRDIMGc_megahr`, `WRDIMGc_mrc` |
| Hypernymy Nouns & Verb | `WRDHYPnv` |



### List of files
* **Benchmark 1**: [wolfe_reports_editorials.csv](https://github.com/phosseini/GisPy/blob/master/data/benchmarks/wolfe_reports_editorials.csv)
* **Benchmark 2**: [wolfe_methods_discussion.csv](https://github.com/phosseini/GisPy/blob/master/data/benchmarks/wolfe_methods_discussion.csv)
* **Benchmark 3**: [Disney](https://github.com/phosseini/GisPy/tree/master/data/benchmarks/disney)
* [`experiments.ipynb`](https://github.com/phosseini/GisPy/blob/master/notebooks/experiments.ipynb): all experiments including the robustness tests on three benchmarks.
* [`benchmarks.ipynb`](https://github.com/phosseini/GisPy/blob/master/notebooks/benchmarks.ipynb): preprocessing Wolfe's benchmark files.

### Gist Inference Score (GIS) formula

```
GIS = Referential Cohesion 
      + Deep Cohesion 
      + (LSA Verb Overlap - WordNet Verb Overlap) 
      - Word Concreteness 
      - Imageability 
      - Hypernymy Nouns & Verbs
```

### Citation
```bibtex
@article{hosseini2022gispy,
  title={GisPy: A Tool for Measuring Gist Inference Score in Text},
  author={Hosseini, Pedram and Wolfe, Christopher R and Diab, Mona and Broniatowski, David A},
  journal={arXiv preprint arXiv:2205.12484},
  year={2022}
}
```
