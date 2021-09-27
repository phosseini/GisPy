import os
import re
import copy
import json
import torch
import itertools
import statistics
import numpy as np
import pandas as pd
import torch.nn as nn

from utils import find_mrc_word
from utils import get_causal_cues
from data_reader import convert_docs

from os import listdir
from scipy.stats import zscore
from os.path import isfile, join
from nltk.corpus import wordnet as wn
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class GIST:
    def __init__(self, docs_path='../data/documents', config_path='gist_config.json'):
        print('loading parameters and models...')
        self.docs_path = docs_path
        # loading parameters
        if os.path.exists(config_path):
            with open(config_path) as f:
                params = json.load(f)
        else:
            raise FileNotFoundError('Please put the config file in the following path: /gist_config.json')

        model_name = params['model_name']
        self.document_batch_size = params['document_batch_size']

        # since documents may be longer than BERT's maximum length, we use Longformer model with maximum length of 4096
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # self.model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')

    def compute_scores(self, min_document_count=2):
        """
        computing the Gist Inference Score (GIS) for a collection of documents
        :param min_document_count: minimum number of documents required to compute the GIS
        :return:
        """

        # ----------------
        # initialize lists
        scores = {}
        normalized_scores = {}
        z_scores = {}
        list_of_indices = ["PCREFz", "PCDCz", "SMCAUSlme", "SMCAUSwn", "WRDCNCc", "WRDIMGc", "WRDHYPnv"]
        for index in list_of_indices:
            scores[index] = []  # raw scores
            normalized_scores[index] = []  # to store normalized scores to 0-1 range
            z_scores[index] = []  # to store z-scores of indices
        # ----------------

        # Step 1: reading the raw text files/documents
        print('reading input text files...')
        if os.path.isdir(self.docs_path):
            # reading text files and building a data frame
            df_docs = pd.DataFrame(columns=["d_id", "text", "gis", "gis_zscore"])
            txt_files = [f for f in listdir(self.docs_path) if isfile(join(self.docs_path, f)) and '.txt' in f]
            for txt_file in txt_files:
                with open('{}/{}'.format(self.docs_path, txt_file), 'r') as input_file:
                    df_docs = df_docs.append({"d_id": txt_file, 'text': input_file.read()}, ignore_index=True)
        else:
            raise Exception(
                'The document directory path you are using does not exist.\nCurrent path: {}'.format(self.docs_path))

        # Step 2: converting the raw text files into tokens and adding their POS tags
        if len(df_docs) < min_document_count:
            raise Exception('There should be minimum two documents to process')
        elif self.document_batch_size > len(df_docs):
            raise Exception(
                'batch size (document_batch_size parameter) should not be greater than the total number of documents.'
                '\nCurrent number of documents: {}\nCurrent batch size: {}'.format(len(df_docs),
                                                                                   self.document_batch_size))

        else:
            batch_counter = 1
            doc_ids = list(set(df_docs['d_id']))
            print('------------------------------')
            print('number of documents: {}'.format(len(df_docs)))
            print('document batch size: {}'.format(self.document_batch_size))
            print('document(s) in each batch: {}'.format(int(len(df_docs) / self.document_batch_size)))
            print('------------------------------')
            error_docs = []
            for batch in np.array_split(df_docs, self.document_batch_size):
                print('processing batch #{}'.format(batch_counter))
                batch_doc_ids = list(set(batch['d_id']))
                df_docs_tokens = convert_docs(batch)

                # -----------------------------------------------
                # computing token and sentence embeddings
                embeddings = self._get_embeddings(df_docs_tokens)
                # -----------------------------------------------
                for doc_id in batch_doc_ids:
                    err_flag = False
                    try:
                        df_doc = df_docs_tokens.loc[df_docs_tokens['d_id'] == doc_id]
                        PCREFz = self._compute_PCREFz(embeddings[doc_id][0])
                        SMCAUSlme = self._compute_SMCAUSlme(df_doc, embeddings[doc_id])
                        _, _, PCDCz = self._find_causal_connectives(df_doc)
                        SMCAUSwn = self._compute_SMCAUSwn(df_doc)
                        WRDCNCc, WRDIMGc = self._compute_WRDCNCc_WRDIMGc(df_doc)
                        WRDHYPnv = self._compute_WRDHYPnv(df_doc)
                    except Exception as e:
                        err_flag = True
                        error_docs.append(doc_id)
                        print('Error in computing the indexes of document: {}'.format(doc_id))
                        print(e)

                    # checking if all the indices are computed without any error
                    if not err_flag:
                        scores["PCREFz"].append(PCREFz)
                        scores["PCDCz"].append(PCDCz)
                        scores["SMCAUSlme"].append(SMCAUSlme)
                        scores["SMCAUSwn"].append(SMCAUSwn)
                        scores["WRDCNCc"].append(WRDCNCc)
                        scores["WRDIMGc"].append(WRDIMGc)
                        scores["WRDHYPnv"].append(WRDHYPnv)

                batch_counter += 1

        # GIS formula
        # gis_score = PCREFz + PCDCz + (SMCAUSlsa - SMCAUSwn) - PCCNCz - zWRDIMGc - WRDHYPnv

        print('normalizing values of indices...')
        for idx_name, idx_values in scores.items():
            # saving the z-score
            z_scores[idx_name] = zscore(idx_values)

            # computing the normalized score in [0, 1] range
            min_val = min(idx_values)
            max_val = max(idx_values)
            for val in idx_values:
                normalized = (val - min_val) / (max_val - min_val)
                normalized_scores[idx_name].append(normalized)

        # computing Gist Inference Score (GIS) for documents
        print('computing the final GIS...')
        score_types = {'gis': normalized_scores, 'gis_zscore': z_scores}
        for i in range(len(doc_ids)):
            if doc_ids[i] not in error_docs:
                # computing different scores for the document
                for score_type, scores_values in score_types.items():
                    doc_score = scores_values["PCREFz"][i] + scores_values["PCDCz"][i] + (
                            scores_values["SMCAUSlme"][i] - scores_values["SMCAUSwn"][i]) - \
                                scores_values["WRDCNCc"][i] - scores_values["WRDIMGc"][i] - \
                                scores_values["WRDHYPnv"][i]
                    df_docs.loc[df_docs['d_id'] == doc_ids[i], score_type] = doc_score

        try:
            # saving result in a csv file
            df_docs.to_csv('results.csv')
            print('computing GIS for all documents is done. results are saved at /results.csv')
        except Exception as e:
            print('error in saving the results. detail: {}'.format(e))

    @staticmethod
    def _get_doc_sentences(df_doc):
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

    def _find_causal_connectives(self, df_doc):
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
        sentences = self._get_doc_sentences(df_doc)
        for sentence in sentences:
            for pattern in patterns:
                if bool(pattern.match(sentence)):
                    causal_connectives_count += 1
                    matched_patterns.append(pattern)
        return causal_connectives_count, matched_patterns, causal_connectives_count / len(sentences)

    def _compute_SMCAUSwn(self, df_doc):
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
                # assigning the lowest score to words for which we can't compute the score
                scores.append(0)
        return sum(scores) / len(scores)

    def _compute_WRDCNCc_WRDIMGc(self, df_doc):
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

    def _compute_SMCAUSlme(self, df_doc, doc_embedding, pos_tags=['VERB']):
        """
        computing the similarity among tokens with certain POS tag in document
        lme stands for the Language Model-based Embedding which is a replacement for Latent Semantic Analysis (LSA) here
        :param df_doc: data frame of a document
        :param pos_tags: list of part-of-speech tags for which we want to compute the cosine similarity
        :return:
        """

        tokens, token_embeddings = doc_embedding[1], doc_embedding[2]

        i = 0
        i_token = 0
        word_embeddings = []
        while i < len(df_doc):
            if df_doc.iloc[i]['token_pos'] in pos_tags:
                # true, if there's no sub-token
                if df_doc.iloc[i]['token_text'].lower() == tokens[i_token].lower():
                    word_embeddings.append(token_embeddings[i_token])
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
                    word_embeddings.append(torch.mean(torch.stack(tensors), dim=0))
                    i += 1
                    i_token = copy.deepcopy(j)
            else:
                i += 1
                i_token += 1

        # checking if we have the embeddings of all VERBs
        assert len(df_doc.loc[df_doc['token_pos'].isin(pos_tags)]) == len(word_embeddings)

        # computing the cosine similarity among all VERBs in document
        scores = []
        cosine = nn.CosineSimilarity(dim=0)
        for pair in itertools.combinations(word_embeddings, r=2):
            scores.append(cosine(pair[0], pair[1]).item())

        return sum(scores) / len(scores)

    def _compute_PCREFz(self, sentence_embeddings):
        """
        Computing Text Easability PC Referential cohesion
        :param df_doc: data frame of a document
        :return:
        """
        scores = []
        for pair in itertools.combinations(sentence_embeddings, r=2):
            a = np.reshape(pair[0], (1, pair[0].size()[0]))
            b = np.reshape(pair[1], (1, pair[1].size()[0]))
            scores.append(cosine_similarity(a, b).item())

        return sum(scores) / len(scores)

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

    @staticmethod
    # Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _get_embeddings(self, df_docs, max_length=128):
        """
        tokenizing a string with BERT and getting the embeddings of each token.
        :param doc:
        :param layers:
        :return:
        """

        # Step 1: creating list of all sentences of all documents for batch tokenization
        all_sentences = []
        doc_sentences_count = []
        doc_ids = list(set(df_docs['d_id']))
        for doc_id in doc_ids:
            doc_sentences = self._get_doc_sentences(df_docs.loc[df_docs['d_id'] == doc_id])
            all_sentences.extend(doc_sentences)
            # keeping track of number of sentences in each document
            doc_sentences_count.append(len(doc_sentences))

        # Step 2: now, tokenization and computing the embeddings
        encoded_input = self.tokenizer(all_sentences, padding=True, truncation=True, max_length=max_length,
                                       return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling to compute sentence embeddings. In this case, mean pooling
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        # get all hidden states
        # size of states: (number of sentences) * (embedding size) * (embedding dimension)
        states = model_output.last_hidden_state

        sentence_idx = 0
        all_embeddings = {}  # to store token and sentence embeddings of all documents
        for i in range(len(doc_ids)):
            all_embeddings[doc_ids[i]] = [[], [], []]
            doc_sentence_embeddings = []
            doc_tokens = []
            doc_token_embeddings = []
            for j in range(sentence_idx, sentence_idx + doc_sentences_count[i]):
                doc_sentence_embeddings.append(sentence_embeddings[j])
                # computing the token embedding
                tokens = [self.tokenizer.decode(idx) for idx in encoded_input['input_ids'][j]]
                token_embeddings = [token_embedding for token_embedding in states[j]]
                tokens = tokens[1:-1]
                token_embeddings = token_embeddings[1:-1]

                assert len(tokens) == len(token_embeddings)

                doc_tokens.extend(tokens)
                doc_token_embeddings.extend(token_embeddings)

            sentence_idx += doc_sentences_count[i]

            # saving sentence embeddings
            all_embeddings[doc_ids[i]][0] = doc_sentence_embeddings
            all_embeddings[doc_ids[i]][1] = doc_tokens
            all_embeddings[doc_ids[i]][2] = doc_token_embeddings

        return all_embeddings


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
