import torch
import copy
import itertools
import statistics
import pandas as pd
import torch.nn as nn

from scipy.stats import zscore
from transformers import AutoModel, AutoTokenizer
from nltk.corpus import wordnet as wn

from utils import find_mrc_word


class GIST:
    def __init__(self, doc):
        self.doc = doc
        self.tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
        self.model = AutoModel.from_pretrained('bert-large-uncased')

    def compute_SMCAUSwn(self):
        """
        computing the WordNet Verb Overlap in a document
        :return:
        """
        # gettling all VERBs in document
        verbs = list(self.doc.loc[self.doc['token_pos'] == 'VERB'].token_lemma)
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

    def compute_WRDHYPnv(self):
        """
        computing the specificity of a word within the WordNet hierarchy
        :return:
        """
        # getting all VERBs and NOUNs in document
        verbs_nouns = self.doc.loc[(self.doc['token_pos'] == 'VERB') | (self.doc['token_pos'] == 'NOUN')][
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

    def compute_WRDCNCc_WRDIMGc(self):
        """
        computing the document concreteness and imageability
        :return:
        """
        conc_score = 0
        img_score = 0
        for index, row in self.doc.iterrows():
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
        return conc_score / len(self.doc), img_score / len(self.doc)

    def compute_SMCAUSlme(self, pos_tags=['VERB']):
        """
        computing the similarity among tokens with certain POS tag in document
        lme stands for the Language Model-based Embedding which is a replacement for Latent Semantic Analysis (LSA) here
        :param pos_tags: list of part-of-speech tags for which we want to compute the cosine similarity
        :return:
        """
        # initializing the data frame with doc data
        df = self.doc

        # creating list of tokens
        doc_context = []
        for index, row in df.iterrows():
            doc_context.append(row['token_text'])

        torch.set_grad_enabled(False)

        doc_string = ' '.join(doc_context)
        tokens = self.tokenizer.tokenize(doc_string)

        # This is not sufficient for the model, as it requires integers as input,
        # not a problem, let's convert tokens to ids.
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Add the required special tokens
        tokens_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_ids)

        # We need to convert to a Deep Learning framework specific format, let's use PyTorch for now.
        tokens_pt = torch.tensor([tokens_ids])

        # Now we're ready to go through BERT with out input
        outputs, pooled = self.model(tokens_pt)

        # creating a list of tokens and their embeddings
        last_hidden_states = outputs[0]
        token_embeddings = []
        i = 0
        while i < len(tokens):
            token_embeddings.append([tokens[i], last_hidden_states[i + 1]])
            i += 1

        assert len(tokens) == len(token_embeddings)

        i = 0
        i_token = 0
        verb_embeddings = []
        while i < len(df):
            if df.iloc[i]['token_pos'] in pos_tags:
                # true, if there's no sub-token
                if df.iloc[i]['token_text'].lower() == token_embeddings[i_token][0].lower():
                    verb_embeddings.append(token_embeddings[i_token][1])
                    i += 1
                    i_token += 1
                # it means that there are sub-tokens
                else:
                    # if you want to check the tokens
                    # print(df.iloc[i]['token_text'], tokens[i_token])
                    tensors = [token_embeddings[i_token][1]]
                    j = copy.deepcopy(i_token) + 1

                    # getting embeddings of all sub-tokens of current token and then computing their mean
                    while j < len(tokens) and '#' in tokens[j]:
                        tensors.append(token_embeddings[j][1])
                        j += 1
                    verb_embeddings.append(torch.mean(torch.stack(tensors), dim=0))
                    i += 1
                    i_token = copy.deepcopy(j)
            else:
                i += 1
                i_token += 1

        # checking if we have the embeddings of all VERBs
        assert len(df.loc[df['token_pos'].isin(pos_tags)]), len(verb_embeddings)

        # computing the cosine similarity among all VERBs in document
        scores = []
        cosine = nn.CosineSimilarity(dim=0)
        for pair in itertools.combinations(verb_embeddings, r=2):
            scores.append(cosine(pair[0], pair[1]).item())

        return statistics.mean(scores)


class GIS:
    def __init__(self):
        pass

    def score(self, df):
        """
        computing Gist Inference Score (GIS) based on the following paper:
        https://link.springer.com/article/10.3758/s13428-019-01284-4
        :param df: a dataframe that contains coh-metrix indices
        :return: the input dataframe with an extra column named "GIS" that stores gist inference score
        """
        # Referential Cohesion (PCREFz)
        # Deep Cohesion (PCDCz)
        # Verb Overlap LSA (SMCAUSlsa)
        # Verb Overlap WordNet (SMCAUSwn)
        # Word Concreteness (PCCNCz)
        # Imageability for Content Words (WRDIMGc)
        # Hypernymy for Nouns and Verbs (WRDHYPnv)

        # computing z-scores
        df["zSMCAUSlsa"] = zscore(df['SMCAUSlsa'])
        df["zSMCAUSwn"] = zscore(df['SMCAUSwn'])
        df["zWRDIMGc"] = zscore(df['WRDIMGc'])
        df["zWRDHYPnv"] = zscore(df['WRDHYPnv'])

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
