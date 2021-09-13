import os
import re
import copy
import torch
import itertools
import statistics
import numpy as np
import pandas as pd
import torch.nn as nn

from utils import get_causal_cues

from scipy.stats import zscore
from transformers import AutoModel, AutoTokenizer
from nltk.corpus import wordnet as wn
from os import listdir
from os.path import isfile, join

from utils import find_mrc_word
from data_reader import convert_docs


class GIST:
    def __init__(self, docs_path='../data/documents'):
        self.docs_path = docs_path

        # since documents may be longer than BERT's maximum length, we use Longformer model with maximum length of 4096
        # TODO: we need to also explore other models for finding vector representation of tokens in documents
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # self.model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.model = AutoModel.from_pretrained("allenai/longformer-base-4096", output_hidden_states=True)

    def compute_scores(self, min_document_count=2):
        """
        computing the Gist Inference Score (GIS) for a collection of documents
        :param min_document_count: minimum number of documents required to compute the GIS
        :return:
        """

        # Step 1: reading the raw text files/documents
        if os.path.isdir(self.docs_path):
            # reading text files and building a data frame
            df_docs = pd.DataFrame(columns=["d_id", "text", "gis"])
            txt_files = [f for f in listdir(self.docs_path) if isfile(join(self.docs_path, f)) and '.txt' in f]
            for txt_file in txt_files:
                with open('{}/{}'.format(self.docs_path, txt_file), 'r') as input_file:
                    df_docs = df_docs.append({"d_id": txt_file, 'text': input_file.read()}, ignore_index=True)
        else:
            return 'The directory path does not exist'

        # Step 2: converting the raw text files into tokens and adding their POS tags
        if len(df_docs) < min_document_count:
            return 'There should be minimum two documents to process'
        else:
            doc_ids = list(set(df_docs['d_id']))
            df_docs_tokens = convert_docs(df_docs)

        scores = {"PCDCz": [], "SMCAUSlme": [], "SMCAUSwn": [], "WRDCNCc": [], "WRDIMGc": [], "WRDHYPnv": []}

        error_docs = []
        for doc_id in doc_ids:
            err_flag = False
            try:
                df_doc = df_docs_tokens.loc[df_docs_tokens['d_id'] == doc_id]
                _, _, PCDCz = self.find_causal_connectives(df_doc)
                SMCAUSlme = self.compute_SMCAUSlme(df_doc)
                SMCAUSwn = self.compute_SMCAUSwn(df_doc)
                WRDCNCc, WRDIMGc = self.compute_WRDCNCc_WRDIMGc(df_doc)
                WRDHYPnv = self.compute_WRDHYPnv(df_doc)
            except Exception as e:
                err_flag = True
                error_docs.append(doc_id)
                print('Error in computing the indexes of document: {}'.format(doc_id))
                print(e)

            if not err_flag:
                scores["PCDCz"].append(PCDCz)
                scores["SMCAUSlme"].append(SMCAUSlme)
                scores["SMCAUSwn"].append(SMCAUSwn)
                scores["WRDCNCc"].append(WRDCNCc)
                scores["WRDIMGc"].append(WRDIMGc)
                scores["WRDHYPnv"].append(WRDHYPnv)

        # gis_score = PCREFz + PCDCz + (SMCAUSlsa - SMCAUSwn) - PCCNCz - zWRDIMGc - WRDHYPnv

        # normalizing scores to 0-1 range
        normalized_scores = {"PCDCz": [], "SMCAUSlme": [], "SMCAUSwn": [], "WRDCNCc": [], "WRDIMGc": [], "WRDHYPnv": []}
        for idx_name, idx_values in scores.items():
            min_val = min(idx_values)
            max_val = max(idx_values)
            for val in idx_values:
                normalized = (val - min_val) / (max_val - min_val)
                normalized_scores[idx_name].append(normalized)

        # computing Gist Inference Score (GIS) for documents
        for i in range(len(doc_ids)):
            if doc_ids[i] not in error_docs:
                doc_gis = normalized_scores["PCDCz"][i] + (
                        normalized_scores["SMCAUSlme"][i] - normalized_scores["SMCAUSwn"][i]) - \
                          normalized_scores["WRDCNCc"][i] - normalized_scores["WRDIMGc"][i] - \
                          normalized_scores["WRDHYPnv"][
                              i]

                # saving the GIS for current document
                df_docs.loc[df_docs['d_id'] == doc_ids[i], 'gis'] = doc_gis

        return df_docs

    def get_doc_sentences(self, df_doc):
        sentences = []
        current_sentence = ""
        current_sentence_idx = 0
        for i in range(len(df_doc)):
            if df_doc.iloc[i]['s_id'] == current_sentence_idx:
                current_sentence += df_doc.iloc[i]['token_text'] + ' '
            else:
                # end of current sentence
                sentences.append(current_sentence.strip())

                # reset variables for the next sentence
                current_sentence = ""
                if i < len(df_doc) - 1:
                    current_sentence_idx = df_doc.iloc[i + 1]['s_id']
        return sentences

    def find_causal_verbs(self, df_doc):
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

    def find_causal_connectives(self, df_doc):
        """
        finding the number of causal connectives in sentences in a document
        :return:
        """
        causal_connectives_count = 0
        matched_patterns = []
        causal_cues = get_causal_cues()
        # compile the patterns
        patterns = []
        for idx, row in causal_cues.iterrows():
            patterns.append(re.compile(r'' + row['cue_regex'] + ''))
        sentences = self.get_doc_sentences(df_doc)
        for sentence in sentences:
            for pattern in patterns:
                if bool(pattern.match(sentence)):
                    causal_connectives_count += 1
                    matched_patterns.append(pattern)
        return causal_connectives_count, matched_patterns, causal_connectives_count / len(sentences)

    def compute_SMCAUSwn(self, df_doc):
        """
        computing the WordNet Verb Overlap in a document
        :return:
        """
        # getting all VERBs in document
        verbs = list(df_doc.loc[df_doc['token_pos'] == 'VERB'].token_lemma)
        verb_synsets = {}

        # getting all synsets to which a verb belongs
        for verb in verbs:
            verb_synsets[verb] = list(set(wn.synsets(verb, wn.VERB)))

        pairs = list(zip(verbs, verbs[1:] + verbs[:1]))
        n_overlaps = 0
        for pair in pairs:
            if set(verb_synsets[pair[0]]) & set(verb_synsets[pair[1]]):
                n_overlaps += 1

        return n_overlaps / len(pairs)

    def compute_WRDHYPnv(self, df_doc):
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
                # assigning the lowest score to words for which we can't compute the score
                scores.append(0)
        return sum(scores) / len(scores)

    def compute_WRDCNCc_WRDIMGc(self, df_doc):
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

    def compute_SMCAUSlme(self, df_doc, pos_tags=['VERB']):
        """
        computing the similarity among tokens with certain POS tag in document
        lme stands for the Language Model-based Embedding which is a replacement for Latent Semantic Analysis (LSA) here
        :param df_doc: data frame of a document
        :param pos_tags: list of part-of-speech tags for which we want to compute the cosine similarity
        :return:
        """

        # creating list of tokens
        doc_context = []
        for index, row in df_doc.iterrows():
            doc_context.append(row['token_text'])

        torch.set_grad_enabled(False)

        tokens, token_embeddings = self._get_longformer_token_embeddings(' '.join(doc_context), layers=[-1])

        i = 0
        i_token = 0
        verb_embeddings = []
        while i < len(df_doc):
            if df_doc.iloc[i]['token_pos'] in pos_tags:
                # true, if there's no sub-token
                if df_doc.iloc[i]['token_text'].lower() == tokens[i_token].lower():
                    verb_embeddings.append(token_embeddings[i_token])
                    i += 1
                    i_token += 1
                # it means that there are sub-tokens (a token is broken down to multiple tokens by tokenizer)
                else:
                    # if you want to check the tokens
                    # print(df.iloc[i]['token_text'], tokens[i_token])
                    tensors = [token_embeddings[i_token]]
                    j = copy.deepcopy(i_token) + 1

                    # getting embeddings of all sub-tokens of current token and then computing their mean
                    while j < len(tokens) and '#' in tokens[j]:
                        tensors.append(token_embeddings[j])
                        j += 1
                    verb_embeddings.append(torch.mean(torch.stack(tensors), dim=0))
                    i += 1
                    i_token = copy.deepcopy(j)
            else:
                i += 1
                i_token += 1

        # checking if we have the embeddings of all VERBs
        assert len(df_doc.loc[df_doc['token_pos'].isin(pos_tags)]), len(verb_embeddings)

        # computing the cosine similarity among all VERBs in document
        scores = []
        cosine = nn.CosineSimilarity(dim=0)
        for pair in itertools.combinations(verb_embeddings, r=2):
            scores.append(cosine(pair[0], pair[1]).item())

        return statistics.mean(scores)

    def _get_bert_token_embeddings(self, doc: str, layers: list):
        """
        tokenizing a string with BERT and getting the embeddings of each token.
        :param doc:
        :param layers:
        :return:
        """
        layers = [-4, -3, -2, -1] if layers is None else layers

        encoded = self.tokenizer.encode_plus(doc, return_tensors="pt")

        tokens = []
        for idx in encoded['input_ids'][0]:
            tokens.append(self.tokenizer.decode(idx))

        tokens = tokens[1:-1]

        with torch.no_grad():
            output = self.model(**encoded)

        # get all hidden states
        states = output.hidden_states

        # stack and sum all requested layers
        # And, we exclude the first and last tensors since they're embeddings of special tokens: [CLS] and [SEP]
        output = torch.stack([states[i] for i in layers]).sum(0).squeeze()[1:-1]

        assert len(output) == len(tokens)

        return tokens, output

    def _get_longformer_token_embeddings(self, doc: str, layers: list):
        """
        tokenizing a string with BERT and getting the embeddings of each token.
        :param doc:
        :param layers:
        :return:
        """
        layers = [-4, -3, -2, -1] if layers is None else layers

        encoded = self.tokenizer.encode_plus(doc, return_tensors="pt")

        tokens = []
        for idx in encoded['input_ids'][0]:
            tokens.append(self.tokenizer.decode(idx))

        tokens = tokens[1:-1]

        with torch.no_grad():
            output = self.model(**encoded)

        # get all hidden states
        states = output.last_hidden_state

        # stack and sum all requested layers
        # And, we exclude the first and last tensors since they're embeddings of special tokens: [CLS] and [SEP]
        output = torch.stack([states[i] for i in layers]).sum(0).squeeze()[1:-1]

        assert len(output) == len(tokens)

        return tokens, output


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
