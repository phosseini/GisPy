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
        indices_cols = ["PCREF", "PCDC", "SMCAUSlme", "SMCAUSwn", "WRDCNCc", "WRDIMGc", "WRDHYPnv"]
        df_cols = ["d_id", "text", "gis", "gis_zscore"]
        df_cols.extend(indices_cols)
        df_docs = pd.DataFrame(columns=df_cols)
        docs_with_errors = []

        if os.path.isdir(self.docs_path):
            txt_files = [f for f in listdir(self.docs_path) if isfile(join(self.docs_path, f)) and '.txt' in f]
            print('total # of documents: {}'.format(len(txt_files)))
            for i, txt_file in enumerate(txt_files):
                with open('{}/{}'.format(self.docs_path, txt_file), 'r') as input_file:
                    doc_text = input_file.read()
                    df_doc, token_embeddings = convert_doc(doc_text)
                    doc_sentences = self._get_doc_sentences(df_doc)
                    sentence_embeddings = list(self.sentence_model.encode(doc_sentences))
                    assert len(sentence_embeddings) == len(doc_sentences)
                    err_flag = False
                    try:
                        PCREF = self._compute_PCREF(sentence_embeddings)
                        SMCAUSlme = self._compute_SMCAUSlme(df_doc, token_embeddings)
                        _, _, PCDC = self._find_causal_connectives(doc_sentences)
                        SMCAUSwn = self._compute_SMCAUSwn(df_doc, similarity_measure='wup')
                        WRDCNCc, WRDIMGc = self._compute_WRDCNCc_WRDIMGc_megahr(df_doc)
                        err_flag = True if WRDCNCc is None and WRDIMGc is None else err_flag
                        WRDHYPnv = self._compute_WRDHYPnv(df_doc)
                        print('file #{} done'.format(i + 1))
                    except Exception as e:
                        err_flag = True
                        docs_with_errors.append(txt_file)

                    # checking if all the indices are computed without any error
                    if not err_flag:
                        df_docs = df_docs.append(
                            {"d_id": txt_file, "text": doc_text, "PCREF": PCREF, "PCDC": PCDC, "SMCAUSlme": SMCAUSlme,
                             "SMCAUSwn": SMCAUSwn, "WRDCNCc": WRDCNCc, "WRDIMGc": WRDIMGc, "WRDHYPnv": WRDHYPnv},
                            ignore_index=True)
        else:
            raise Exception(
                'The document directory path you are using does not exist.\nCurrent path: {}'.format(self.docs_path))

        # GIS formula
        # gis_score = PCREFz + PCDCz + (SMCAUSlsa - SMCAUSwn) - PCCNCz - zWRDIMGc - WRDHYPnv
        normalized_scores = dict()
        for col in indices_cols:
            normalized_scores[col] = list()
        z_scores = dict()
        print('normalizing values of indices...')
        if len(df_docs) != 0:
            for idx_name in indices_cols:
                # saving the z-score
                z_scores[idx_name] = zscore(list(df_docs[idx_name]))
                # computing the normalized score in [0, 1] range
                normalized_scores[idx_name] = [float(i) / sum(list(df_docs[idx_name])) for i in list(df_docs[idx_name])]
        else:
            raise Exception(
                'there is no score computed to normalize. check the /error_log.txt to see the detailed errors')

        # computing Gist Inference Score (GIS) for documents
        print('computing the final GIS...')
        score_types = {'gis': normalized_scores, 'gis_zscore': z_scores}
        # we define d_idx since number of all documents may not be same (due to errors)
        # as documents for which we have computed scores
        d_idx = 0
        for i, txt_file in enumerate(txt_files):
            if txt_file not in docs_with_errors:
                # computing different scores for the document
                for score_type, scores in score_types.items():
                    doc_score = scores["PCREF"][d_idx] + scores["PCDC"][d_idx] + (
                            scores["SMCAUSlme"][d_idx] - scores["SMCAUSwn"][d_idx]) - scores["WRDCNCc"][d_idx] - \
                                scores["WRDIMGc"][d_idx] - scores["WRDHYPnv"][d_idx]
                    df_docs.loc[df_docs['d_id'] == txt_file, score_type] = doc_score
                d_idx += 1

        # filtering out rows for which we don't have GIS
        df_docs = df_docs.loc[df_docs['gis'].notnull()]

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

    def score(self, df, wolfe=False):
        """
        computing Gist Inference Score (GIS) based on the following paper:
        https://link.springer.com/article/10.3758/s13428-019-01284-4
        :param df: a dataframe that contains coh-metrix indices
        :param wolfe: whether using wolfe's mean and standard deviation for computing z-score
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

        def z_score(df_col, params):
            if wolfe:
                return df_col.map(lambda x: (x - params['mean']) / params['sd'])
            else:
                return zscore(df_col)

        # computing z-scores
        df["zSMCAUSlsa"] = z_score(df['SMCAUSlsa'], self.wolfe_mean_sd['SMCAUSlsa'])
        df["zSMCAUSwn"] = z_score(df['SMCAUSwn'], self.wolfe_mean_sd['SMCAUSwn'])
        df["zWRDIMGc"] = z_score(df['WRDIMGc'], self.wolfe_mean_sd['WRDIMGc'])
        df["zWRDHYPnv"] = z_score(df['WRDHYPnv'], self.wolfe_mean_sd['WRDHYPnv'])

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
            df.loc[idx, "GIS"] = gis

        return df
