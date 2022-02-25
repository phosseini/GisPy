import os
import re
import copy
import json
import torch
import itertools
# import statistics
import numpy as np
import pandas as pd
import numbers
import torch.nn as nn

from utils import find_mrc_word
from utils import get_causal_cues
from utils import read_megahr_concreteness_imageability
from data_reader import convert_doc

from os import listdir
from scipy.stats import zscore
from os.path import isfile, join
from nltk.corpus import wordnet as wn
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity


class GIST:
    def __init__(self, docs_path='../data/documents'):
        print('loading parameters and models...')
        self.docs_path = docs_path

        # loading parameters
        config_path = 'gist_config.json'
        if os.path.exists(config_path):
            with open(config_path) as f:
                params = json.load(f)
        else:
            raise FileNotFoundError('Please put the config file in the following path: /gist_config.json')

        # reading megahr
        self.megahr_dict = read_megahr_concreteness_imageability()
        self.all_synsets = {}

        # compile the causal patterns
        causal_cues = get_causal_cues()
        self.causal_patterns = []
        for idx, row in causal_cues.iterrows():
            self.causal_patterns.append(re.compile(r'' + row['cue_regex'].lower() + ''))

        self.sentence_model = SentenceTransformer(params['sentence_transformers_model'])

    def compute_scores(self):
        """
        computing the Gist Inference Score (GIS) for a collection of documents
        :return:
        """
        indices_cols = ["PCREF", "PCDC", "SMCAUSlsa", "SMCAUSwn", "PCCNC", "WRDIMGc", "WRDHYPnv"]
        df_cols = ["d_id", "text"]
        df_cols.extend(indices_cols)
        df_docs = pd.DataFrame(columns=df_cols)
        docs_with_errors = []

        if os.path.isdir(self.docs_path):
            txt_files = [f for f in listdir(self.docs_path) if isfile(join(self.docs_path, f)) and '.txt' in f]
            print('total # of documents: {}'.format(len(txt_files)))
            print('computing indices for documents...')
            for i, txt_file in enumerate(txt_files):
                with open('{}/{}'.format(self.docs_path, txt_file), 'r') as input_file:
                    doc_text = input_file.read()
                    df_doc, token_embeddings = convert_doc(doc_text)
                    doc_sentences = self._get_doc_sentences(df_doc)
                    sentence_embeddings = list(self.sentence_model.encode(doc_sentences))
                    assert len(sentence_embeddings) == len(doc_sentences)
                    try:
                        PCREF = self._compute_PCREF(sentence_embeddings)
                        SMCAUSlme = self._compute_SMCAUSlme(df_doc, token_embeddings)
                        _, _, PCDC = self._find_causal_connectives(doc_sentences)
                        SMCAUSwn = self._compute_SMCAUSwn(df_doc, similarity_measure='wup')
                        WRDCNCc, WRDIMGc = self._compute_WRDCNCc_WRDIMGc_megahr(df_doc)
                        WRDHYPnv = self._compute_WRDHYPnv(df_doc)
                        print('#{} done'.format(i + 1))
                        df_docs = df_docs.append(
                            {"d_id": txt_file, "text": doc_text, "PCREF": PCREF, "PCDC": PCDC, "SMCAUSlsa": SMCAUSlme,
                             "SMCAUSwn": SMCAUSwn, "PCCNC": WRDCNCc, "WRDIMGc": WRDIMGc, "WRDHYPnv": WRDHYPnv},
                            ignore_index=True)
                    except Exception as e:
                        docs_with_errors.append(txt_file)

        else:
            raise Exception(
                'The document directory path you are using does not exist.\nCurrent path: {}'.format(self.docs_path))

        print('computing indices for documents is done.')
        print('# of documents with error: {}'.format(len(docs_with_errors)))

        return df_docs

    @staticmethod
    def _get_sentences_count(df_doc):
        """
        get the count of all sentences in a document
        :param df_doc:
        :return:
        """
        sents_count = 0
        paragraph_ids = df_doc['p_id'].unique()
        for p_id in paragraph_ids:
            paragraph_df = df_doc.loc[df_doc['p_id'] == p_id]
            paragraph_sents_count = len(paragraph_df['s_id'].unique())
            sents_count += paragraph_sents_count
        return sents_count

    def _get_doc_sentences(self, df_doc):
        """
        get list of sentences in a document
        :param df_doc:
        :return:
        """
        sentences = list()
        current_sentence = str()
        df_doc.reset_index()
        p_ids = df_doc['p_id'].unique()
        for p_id in p_ids:
            df_paragraph = df_doc.loc[df_doc['p_id'] == p_id]
            current_s_id = 0
            for idx, row in df_paragraph.iterrows():
                if row['s_id'] == current_s_id:
                    current_sentence += row['token_text'] + ' '
                else:
                    # end of current sentence, save it first
                    sentences.append(current_sentence.strip())
                    # reset variables for the next sentence
                    current_sentence = ""
                    current_s_id += 1
            # saving the last sentence
            sentences.append(current_sentence.strip())
        assert len(sentences) == self._get_sentences_count(df_doc)
        return sentences

    @staticmethod
    def _find_causal_verbs(df_doc):
        causal_verbs = []
        verbs = list(df_doc.loc[df_doc['token_pos'] == 'VERB'].token_lemma)
        for verb in verbs:
            verb_synsets = list(set(wn.synsets(verb, wn.VERB)))
            for verb_synset in verb_synsets:
                # checking if this verb can cause or entail anything
                if len(verb_synset.causes()) > 0 or len(verb_synset.entailments()) > 0:
                    causal_verbs.append(verb)
                    break  # we break here since for now we only want to know whether this verb can be causal at all.
        return causal_verbs, len(causal_verbs) / len(verbs)

    def _find_causal_connectives(self, sentences):
        """
        finding the number of causal connectives in sentences in a document
        :return:
        """
        causal_connectives_count = 0
        matched_patterns = list()
        for sentence in sentences:
            for pattern in self.causal_patterns:
                if bool(pattern.match(sentence.lower())):
                    causal_connectives_count += 1
                    matched_patterns.append(pattern)
        return causal_connectives_count, matched_patterns, causal_connectives_count / len(sentences)

    def _compute_SMCAUSwn_v1(self, df_doc):
        """
        computing the WordNet Verb Overlap in a document
        :return:
        """
        # getting all unique VERBs in a document
        verbs = set(list(df_doc.loc[df_doc['token_pos'] == 'VERB'].token_lemma))
        verb_synsets = {}

        # getting all synsets (synonym sets) to which a verb belongs
        for verb in verbs:
            verb_synsets[verb] = set(wn.synsets(verb, wn.VERB))

        n_overlaps = 0
        if len(verbs) > 1:
            pairs = set(list(itertools.combinations(verbs, r=2)))
            for pair in pairs:
                if verb_synsets[pair[0]] & verb_synsets[pair[1]]:
                    n_overlaps += 1
        else:
            return 1

        return n_overlaps / len(pairs)

    def _compute_SMCAUSwn(self, df_doc, similarity_measure='path'):
        """
        computing the WordNet Verb Overlap in a document
        :param similarity_measure: the type of similarity to use, one of the following: ['path', 'lch', 'wup]
        :return:
        """
        # getting all VERBs in a document
        verbs = list(df_doc.loc[df_doc['token_pos'] == 'VERB'].token_lemma)
        verb_synsets = dict()

        # getting all synsets (synonym sets) to which a verb belongs
        for verb in verbs:
            # check if synset is already in dictionary to avoid calling WordNet
            if verb in self.all_synsets:
                verb_synsets[verb] = self.all_synsets[verb]
            else:
                synsets = set(wn.synsets(verb, wn.VERB))
                self.all_synsets[verb] = synsets
                verb_synsets[verb] = synsets

        verb_pairs = itertools.combinations(verbs, r=2)

        similarity_scores = list()

        # computing the similarity of verb pairs by computing the average similarity between their synonym sets
        # each verb can have one or multiple synonym sets
        for verb_pair in verb_pairs:
            synset_pairs = itertools.product(verb_synsets[verb_pair[0]], verb_synsets[verb_pair[1]])
            for synset_pair in synset_pairs:
                if similarity_measure == 'path':
                    similarity_score = wn.path_similarity(synset_pair[0], synset_pair[1])
                elif similarity_measure == 'lch':
                    similarity_score = wn.lch_similarity(synset_pair[0], synset_pair[1])
                elif similarity_measure == 'wup':
                    similarity_score = wn.wup_similarity(synset_pair[0], synset_pair[1])

                # check if similarity_score is not None and is a number
                if isinstance(similarity_score, numbers.Number):
                    similarity_scores.append(similarity_score)

        return sum(similarity_scores) / len(similarity_scores)

    def _compute_WRDHYPnv(self, df_doc):
        """
        computing the specificity of a word within the WordNet hierarchy
        :return:
        """
        # getting all VERBs and NOUNs in document
        verbs_nouns = df_doc.loc[(df_doc['token_pos'] == 'VERB') | (df_doc['token_pos'] == 'NOUN')][
            ['token_text', 'token_lemma', 'token_pos']]

        scores = []
        for index, row in verbs_nouns.iterrows():
            try:
                if row['token_pos'] == 'VERB':
                    synsets = list(set(wn.synsets(row['token_lemma'], wn.VERB)))
                elif row['token_pos'] == 'NOUN':
                    synsets = list(set(wn.synsets(row['token_lemma'], wn.NOUN)))

                hypernym_path_length = 0
                for synset in synsets:
                    hypernym_path_length += len(synset.hypernym_paths())
                # computing the average length of hypernym path
                hypernym_score = hypernym_path_length / len(synsets)
                scores.append(hypernym_score)
            except:
                pass
        return sum(scores) / len(scores)

    def _compute_WRDCNCc_WRDIMGc_mrc(self, df_doc):
        """
        computing the document concreteness and imageability
        :return:
        """
        conc_score = 0
        img_score = 0
        for index, row in df_doc.iterrows():
            records = find_mrc_word(row['token_text'], row['token_pos'])

            # there might be more than one record for the very word with its POS tag
            if len(records) > 0:
                word_conc_score = 0
                word_img_score = 0
                for record in records:
                    word_conc_score += record.conc
                    word_img_score += record.imag
                conc_score += (word_conc_score / len(records))
                img_score += (word_img_score / len(records))
        return conc_score / len(df_doc), img_score / len(df_doc)

    def _compute_WRDCNCc_WRDIMGc_megahr(self, df_doc):
        """
        computing the document concreteness and imageability
        :return:
        """
        concreteness_scores = list()
        imageability_scores = list()

        # filtering out tokens we don't need
        pos_filter = ['NUM', 'PUNCT', 'SYM']
        df_doc = df_doc.loc[~df_doc['token_pos'].isin(pos_filter)]

        for index, row in df_doc.iterrows():
            token_text = row['token_text'].lower()
            if token_text in self.megahr_dict:
                concreteness_scores.append(self.megahr_dict[token_text][0])
                imageability_scores.append(self.megahr_dict[token_text][1])

        if len(concreteness_scores) > 0 and len(imageability_scores) > 0:
            return sum(concreteness_scores) / len(concreteness_scores), sum(imageability_scores) / len(
                imageability_scores)
        else:
            return None, None

    def _compute_SMCAUSlme(self, df_doc, token_embeddings, pos_tags=['VERB']):
        """
        computing the similarity among tokens with certain POS tag in a document
        lme stands for the Language Model-based Embedding which is a replacement for Latent Semantic Analysis (LSA) here
        :param df_doc: data frame of a document
        :param pos_tags: list of part-of-speech tags for which we want to compute the cosine similarity
        :return:
        """
        scores = list()
        word_embeddings = list()
        for idx, row in df_doc.iterrows():
            if row['token_pos'] in pos_tags:
                word_embeddings.append(token_embeddings[row['u_id']])

        # double-checking if we have the embeddings of tokens with the specific POS tags
        assert len(df_doc.loc[df_doc['token_pos'].isin(pos_tags)]) == len(word_embeddings)

        # computing the cosine similarity among all VERBs in document
        if len(word_embeddings) > 1:
            pairs = itertools.combinations(word_embeddings, r=2)
            for pair in pairs:
                scores.append(util.cos_sim(pair[0], pair[1]).item())
        else:
            # if there's only one token in the document (which shouldn't often happen), then there's %100 similarity
            return 1

        return sum(scores) / len(scores)

    def _compute_PCREF(self, sentence_embeddings):
        """
        Computing Text Easability PC Referential cohesion
        :param sentence_embeddings: embeddings of sentences in a document
        :return:
        """
        scores = []  # sentence transformers cosine similarity scores
        if len(sentence_embeddings) > 1:
            pairs = itertools.combinations(sentence_embeddings, r=2)
            for pair in pairs:
                scores.append((util.cos_sim(pair[0], pair[1]).item()))
        else:
            # if there is only one sentence in the document, then there is %100 referential cohesion
            return 1

        return sum(scores) / len(scores)


class GIS:
    def __init__(self):
        self.wolfe_mean_sd = {'SMCAUSlsa': {'mean': 0.097, 'sd': 0.04},
                              'SMCAUSwn': {'mean': 0.553, 'sd': 0.096},
                              'WRDIMGc': {'mean': 410.346, 'sd': 24.994},
                              'WRDHYPnv': {'mean': 1.843, 'sd': 0.26}}

    def _z_score(self, df, index_name, wolfe=False):
        if wolfe:
            params = self.wolfe_mean_sd[index_name]
            return df[index_name].map(lambda x: (x - params['mean']) / params['sd'])
        else:
            return zscore(df[index_name])

    def score(self, df, wolfe=False, gispy=False):
        """
        computing Gist Inference Score (GIS) based on the following paper:
        https://link.springer.com/article/10.3758/s13428-019-01284-4
        use this method when values of indices are computed using CohMetrix
        :param df: a dataframe that contains coh-metrix indices
        :param wolfe: whether using wolfe's mean and standard deviation for computing z-score
        :param gispy: whether indices are computed by gispy or not (if not gispy, indices should be computed by CohMetrix)
        :return: the input dataframe with an extra column named "GIS" that stores gist inference score
        """

        # Referential Cohesion (PCREFz)
        # Deep Cohesion (PCDCz)
        # Verb Overlap LSA (SMCAUSlsa)
        # Verb Overlap WordNet (SMCAUSwn)
        # Word Concreteness (PCCNCz)
        # Imageability for Content Words (WRDIMGc)
        # Hypernymy for Nouns and Verbs (WRDHYPnv)

        # Z-Score(X) = (X-μ)/σ
        # X: a single raw data value
        # μ: population mean
        # σ: population standard deviation

        # computing z-scores
        df["zSMCAUSlsa"] = self._z_score(df, index_name='SMCAUSlsa', wolfe=wolfe)
        df["zSMCAUSwn"] = self._z_score(df, index_name='SMCAUSwn', wolfe=wolfe)
        df["zWRDIMGc"] = self._z_score(df, index_name='WRDIMGc', wolfe=wolfe)
        df["zWRDHYPnv"] = self._z_score(df, index_name='WRDHYPnv', wolfe=wolfe)

        if gispy:
            # since wolfe doesn't have mean and sd for the following indices, we go with wolfe=False here
            df["PCREFz"] = self._z_score(df, index_name='PCREF')
            df["PCDCz"] = self._z_score(df, index_name='PCDC')
            df["PCCNCz"] = self._z_score(df, index_name='PCCNC')

        # computing the Gist Inference Score (GIS)
        for idx, row in df.iterrows():
            PCREFz = row["PCREFz"]
            PCDCz = row["PCDCz"]
            PCCNCz = row["PCCNCz"]
            zSMCAUSlsa = row["zSMCAUSlsa"]
            zSMCAUSwn = row["zSMCAUSwn"]
            zWRDIMGc = row["zWRDIMGc"]
            zWRDHYPnv = row["zWRDHYPnv"]
            gis = PCREFz + PCDCz + (zSMCAUSlsa - zSMCAUSwn) - PCCNCz - zWRDIMGc - zWRDHYPnv
            df.loc[idx, "gis"] = gis

        return df
