import os
import re
import json
import numbers
import itertools
import statistics
import pandas as pd

from os import listdir
import scipy.stats as stats
from os.path import isfile, join
from nltk.corpus import wordnet as wn
from stanza.server import CoreNLPClient
from sentence_transformers import SentenceTransformer, util

from utils import find_mrc_word
from utils import get_causal_cues
from utils import read_megahr_concreteness_imageability
from data_reader import convert_doc


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
        self.verb_similarity = dict()

        # compile the causal patterns
        causal_cues = get_causal_cues()
        self.causal_patterns = list()
        for idx, row in causal_cues.iterrows():
            self.causal_patterns.append(re.compile(r'' + row['cue_regex'].lower() + ''))

        self.sentence_model = SentenceTransformer(params['sentence_transformers_model'])

    @staticmethod
    def _local_cosine(embeddings):
        """
        computing cosine only for consecutive sentence embeddings
        :param embeddings:
        :return:
        """
        if len(embeddings) <= 1:
            return 0
        else:
            i = 0
            scores = list()
            while i + 1 < len(embeddings):
                scores.append(util.cos_sim(embeddings[i], embeddings[i + 1]).item())
                i += 1
            return statistics.mean(scores)

    @staticmethod
    def _global_cosine(embeddings):
        """
        computing cosine of all pairs of sentence embeddings
        :param embeddings:
        :return:
        """
        scores = list()
        pairs = itertools.combinations(embeddings, r=2)
        for pair in pairs:
            scores.append((util.cos_sim(pair[0], pair[1]).item()))
        return statistics.mean(scores) if len(scores) > 0 else 0

    @staticmethod
    def _clean_text(text):
        encoded_text = text.encode("ascii", "ignore")
        text = encoded_text.decode()
        text = text.replace('…', '...')
        text = re.sub(' +', ' ', text)
        text = re.sub(r'\n+', '\n', text).strip()
        return text

    def compute_scores(self):
        """
        computing the Gist Inference Score (GIS) for a collection of documents
        :return:
        """
        indices_cols = ["DESPC", "DESSC", "CoREF", "PCREF1", "PCREFa", "PCREF1p", "PCREFap", "PCDC", "SMCAUSe_1",
                        "SMCAUSe_a", "SMCAUSe_1p", "SMCAUSe_ap",
                        "SMCAUSwn_1p_path", "SMCAUSwn_1p_lch", "SMCAUSwn_1p_wup",
                        "SMCAUSwn_ap_path", "SMCAUSwn_ap_lch", "SMCAUSwn_ap_wup",
                        "SMCAUSwn_1_path", "SMCAUSwn_1_lch", "SMCAUSwn_1_wup",
                        "SMCAUSwn_a_path", "SMCAUSwn_a_lch", "SMCAUSwn_a_wup",
                        "PCCNC", "WRDIMGc", "WRDHYPnv"]
        df_cols = ["d_id", "text"]
        df_cols.extend(indices_cols)
        df_docs = pd.DataFrame(columns=df_cols)
        docs_with_errors = list()
        with CoreNLPClient(
                annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'coref'],
                threads=10,
                timeout=60000,
                memory='20G',
                be_quiet=True) as client:
            if os.path.isdir(self.docs_path):
                txt_files = [f for f in listdir(self.docs_path) if isfile(join(self.docs_path, f)) and '.txt' in f]
                print('total # of documents: {}'.format(len(txt_files)))
                print('computing indices for documents...')
                for i, txt_file in enumerate(txt_files):
                    with open('{}/{}'.format(self.docs_path, txt_file), 'r') as input_file:
                        doc_text = input_file.read()
                        doc_text = self._clean_text(doc_text)
                        df_doc, token_embeddings = convert_doc(doc_text)
                        doc_sentences, n_paragraphs, n_sentences = self._get_doc_sentences(df_doc)
                        token_ids_by_sentence = self._get_doc_token_ids_by_sentence(df_doc)
                        # -------------------------------
                        # finding the coref using corenlp
                        coref_scores = list()
                        for p_id, p_sentences in doc_sentences.items():
                            paragraph_text = ' '.join(p_sentences)
                            ann = client.annotate(paragraph_text)
                            chain_count = len(list(ann.corefChain))
                            coref_score = chain_count / len(p_sentences)
                            coref_scores.append(coref_score)
                        CoREF = statistics.mean(coref_scores)
                        # -------------------------------
                        sentence_embeddings = dict()
                        all_sentences = list()
                        # initializing sentences and embeddings list
                        for p_id, sentences in doc_sentences.items():
                            all_sentences.extend(sentences)
                            sentence_embeddings[p_id] = [0] * len(sentences)
                        # computing all embeddings at once
                        all_embeddings = list(self.sentence_model.encode(all_sentences))
                        s_index = 0
                        for p_id, sentences in doc_sentences.items():
                            for idx, sentence in enumerate(sentences):
                                if sentence == all_sentences[s_index]:
                                    sentence_embeddings[p_id][idx] = all_embeddings[s_index]
                                    s_index += 1
                        try:
                            PCREF1, PCREFa, PCREF1p, PCREFap = self._compute_PCREF(sentence_embeddings)
                            SMCAUSe_1, SMCAUSe_a, SMCAUSe_1p, SMCAUSe_ap = self._compute_SMCAUSe(df_doc,
                                                                                                 token_embeddings,
                                                                                                 token_ids_by_sentence)
                            SMCAUSwn = self._compute_SMCAUSwn(df_doc, token_ids_by_sentence)
                            _, _, PCDC = self._find_causal_connectives(doc_sentences)
                            WRDCNCc, WRDIMGc = self._compute_WRDCNCc_WRDIMGc_megahr(df_doc)
                            WRDHYPnv = self._compute_WRDHYPnv(df_doc)
                            print('#{} done'.format(i + 1))
                            df_docs = df_docs.append(
                                {"d_id": txt_file, "text": doc_text, "DESPC": n_paragraphs, "DESSC": n_sentences,
                                 "CoREF": CoREF, "PCREF1": PCREF1, "PCREFa": PCREFa, "PCREF1p": PCREF1p,
                                 "PCREFap": PCREFap, "PCDC": PCDC,
                                 "SMCAUSe_1": SMCAUSe_1, "SMCAUSe_a": SMCAUSe_a, "SMCAUSe_1p": SMCAUSe_1p,
                                 "SMCAUSe_ap": SMCAUSe_ap,
                                 'SMCAUSwn_1p_path': SMCAUSwn['SMCAUSwn_1p_path'],
                                 'SMCAUSwn_1p_lch': SMCAUSwn['SMCAUSwn_1p_lch'],
                                 'SMCAUSwn_1p_wup': SMCAUSwn['SMCAUSwn_1p_wup'],
                                 'SMCAUSwn_ap_path': SMCAUSwn['SMCAUSwn_ap_path'],
                                 'SMCAUSwn_ap_lch': SMCAUSwn['SMCAUSwn_ap_lch'],
                                 'SMCAUSwn_ap_wup': SMCAUSwn['SMCAUSwn_ap_wup'],
                                 'SMCAUSwn_1_path': SMCAUSwn['SMCAUSwn_1_path'],
                                 'SMCAUSwn_1_lch': SMCAUSwn['SMCAUSwn_1_lch'],
                                 'SMCAUSwn_1_wup': SMCAUSwn['SMCAUSwn_1_wup'],
                                 'SMCAUSwn_a_path': SMCAUSwn['SMCAUSwn_a_path'],
                                 'SMCAUSwn_a_lch': SMCAUSwn['SMCAUSwn_a_lch'],
                                 'SMCAUSwn_a_wup': SMCAUSwn['SMCAUSwn_a_wup'],
                                 "PCCNC": WRDCNCc, "WRDIMGc": WRDIMGc, "WRDHYPnv": WRDHYPnv},
                                ignore_index=True)
                        except Exception as e:
                            docs_with_errors.append(txt_file)
                            print(e)
            else:
                raise Exception(
                    'The document directory path you are using does not exist.\nCurrent path: {}'.format(
                        self.docs_path))

        print('computing indices for documents is done.')
        print('# of documents with error: {}'.format(len(docs_with_errors)))

        return df_docs

    @staticmethod
    def _filter_tokens_by_pos(df_doc, token_ids_by_sentence, pos_tags: list):
        tokens = dict()
        # keys: paragraph ids
        # values: one list of sentence embeddings for each paragraph id
        for p_s_id, token_ids in token_ids_by_sentence.items():
            p_id = p_s_id.split('_')[0]
            if p_id not in tokens:
                tokens[p_id] = list()
            current_tokens = list()
            for u_id in token_ids:
                row = df_doc.loc[df_doc['u_id'] == u_id]
                if row.iloc[0]['token_pos'] in pos_tags:
                    current_tokens.append({u_id: row.iloc['token_text']})
            if len(current_tokens) > 0:
                tokens[p_id].append(current_tokens)
        return tokens

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
        sentences = dict()
        current_sentence = str()
        df_doc.reset_index()
        p_ids = df_doc['p_id'].unique()
        for p_id in p_ids:
            sentences[p_id] = list()
            df_paragraph = df_doc.loc[df_doc['p_id'] == p_id]
            current_s_id = 0
            for idx, row in df_paragraph.iterrows():
                if row['s_id'] == current_s_id:
                    current_sentence += row['token_text'] + ' '
                else:
                    # end of current sentence, save it first
                    sentences[p_id].append(current_sentence.strip())
                    # reset variables for the next sentence
                    current_sentence = row['token_text'] + ' '
                    current_s_id += 1
            # saving the last sentence
            sentences[p_id].append(current_sentence.strip())
        len_sentences = sum([len(sentences[pid]) for pid in sentences.keys()])
        assert len_sentences == self._get_sentences_count(df_doc)
        return sentences, len(p_ids), len_sentences

    def _get_doc_token_ids_by_sentence(self, df_doc):
        """
        get list of token ids of sentences in a document
        :param df_doc:
        :return:
        """
        sentences_tokens = dict()
        df_doc.reset_index()
        p_ids = df_doc['p_id'].unique()
        for p_id in p_ids:
            current_sentence = list()
            df_paragraph = df_doc.loc[df_doc['p_id'] == p_id]
            current_s_id = 0
            for idx, row in df_paragraph.iterrows():
                if row['s_id'] == current_s_id:
                    current_sentence.append(row['u_id'])
                else:
                    # end of current sentence, save it first
                    sentences_tokens['{}_{}'.format(p_id, current_s_id)] = current_sentence
                    # reset variables for the next sentence
                    current_sentence = [row['u_id']]
                    current_s_id += 1
            # saving the last sentence
            sentences_tokens['{}_{}'.format(p_id, current_s_id)] = current_sentence
        tokens_count = sum([len(v) for k, v in sentences_tokens.items()])
        assert tokens_count == len(df_doc)
        return sentences_tokens

    @staticmethod
    def _find_causal_verbs(df_doc):
        causal_verbs = list()
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
        n_causal_connectives = 0
        matched_patterns = list()
        for p_id, p_sentences in sentences.items():
            for sentence in p_sentences:
                for pattern in self.causal_patterns:
                    if bool(pattern.match(sentence.lower())):
                        n_causal_connectives += 1
                        matched_patterns.append(pattern)
        sentences_count = sum([len(sentences[p_id]) for p_id in sentences.keys()])
        return n_causal_connectives, matched_patterns, n_causal_connectives / sentences_count

    def _compute_SMCAUSwn_v1(self, df_doc, similarity_measure='path'):
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

        return statistics.mean(similarity_scores)

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

    def _compute_SMCAUSe(self, df_doc, token_embeddings, token_ids_by_sentence, pos_tags=['VERB']):
        """
        computing the similarity among tokens with certain POS tag in a document
        *e* at the end stands for Embedding to show this method is a replacement for Latent Semantic Analysis (LSA) here
        :param df_doc: data frame of a document
        :param pos_tags: list of part-of-speech tags for which we want to compute the cosine similarity
        :return:
        """
        # we use this dictionary to avoid computing cosine for the very pair multiple times
        # by simply caching the cosine value of token pairs
        tokens_similarity = dict()

        def local_cosine(e):
            """
            e is a list of embeddings
            :param e:
            :return:
            """
            if len(e) <= 1:
                return 0
            else:
                j = 0
                scores = list()
                while j + 1 < len(e):
                    pair_id = '{}@{}'.format(list(e[j].keys())[0], list(e[j + 1].keys())[0])
                    if pair_id not in tokens_similarity:
                        score = util.cos_sim(list(e[j].values())[0], list(e[j + 1].values())[0]).item()
                        tokens_similarity[pair_id] = score
                    else:
                        score = tokens_similarity[pair_id]
                    scores.append(score)
                    j += 1
                return statistics.mean(scores)

        def global_cosine(pairs):
            scores = list()
            for pair in pairs:
                pair_id = '{}@{}'.format(list(pair[0].keys())[0], list(pair[1].keys())[0])
                if pair_id not in tokens_similarity:
                    score = util.cos_sim(list(pair[0].values())[0], list(pair[1].values())[0]).item()
                    tokens_similarity[pair_id] = score
                else:
                    score = tokens_similarity[pair_id]
                scores.append(score)
            return statistics.mean(scores) if len(scores) > 0 else 0

        embeddings = dict()
        # keys: paragraph ids
        # values: one list of sentence embeddings for each paragraph id
        for p_s_id, token_ids in token_ids_by_sentence.items():
            p_id = p_s_id.split('_')[0]
            if p_id not in embeddings:
                embeddings[p_id] = list()
            current_embeddings = list()
            for u_id in token_ids:
                row = df_doc.loc[df_doc['u_id'] == u_id]
                if row.iloc[0]['token_pos'] in pos_tags:
                    current_embeddings.append({u_id: token_embeddings[u_id]})
            if len(current_embeddings) > 0:
                embeddings[p_id].append(current_embeddings)

        scores_1p = list()
        scores_ap = list()

        token_embeddings_flat = list()

        for p_id, s_embeddings in embeddings.items():
            # *** consecutive cosine ***
            if len(s_embeddings) <= 1:
                scores_1p.append(0)
            else:
                i = 0
                while i + 1 < len(s_embeddings):
                    all_pairs = list(itertools.product(s_embeddings[i], s_embeddings[i + 1]))
                    scores_1p.append(global_cosine(all_pairs))
                    i += 1

            # *** global cosine ***
            t_embeddings = list()  # all token embeddings of all tokens in one paragraph
            for item in s_embeddings:
                t_embeddings.extend(item)
                token_embeddings_flat.extend(item)
            all_pairs = itertools.combinations(t_embeddings, r=2)
            scores_ap.append(global_cosine(all_pairs))

        SMCAUSe_1p = statistics.mean(scores_1p)
        SMCAUSe_ap = statistics.mean(scores_ap)

        # computing global and local indices ignoring the paragraphs
        all_pairs = itertools.combinations(token_embeddings_flat, r=2)
        SMCAUSe_a = global_cosine(all_pairs)
        SMCAUSe_1 = local_cosine(token_embeddings_flat)

        return SMCAUSe_1, SMCAUSe_a, SMCAUSe_1p, SMCAUSe_ap

    def _compute_SMCAUSwn(self, df_doc, token_ids_by_sentence, pos_tags=['VERB']):
        """
        computing WordNet Verb Overlap
        :param df_doc: data frame of a document
        :param token_ids_by_sentence:
        :param pos_tags: list of part-of-speech tags for which we want to compute the overlap
        :return:
        """

        # TODO: this method is the main bottleneck in the pipeline. One reason is that for one verb we have too many synsets
        # TODO: we need to figure out how to find the most relevant synsets for a verb
        scores_functions = {'path': wn.path_similarity, 'lch': wn.lch_similarity, 'wup': wn.wup_similarity}

        def synset_pair_similarity(pair):
            scores = {'path': list(), 'lch': list(), 'wup': list()}
            verb_pair = '{}@{}'.format(list(pair[0].keys())[0], list(pair[1].keys())[0])
            if verb_pair in self.verb_similarity:
                for score_name in ['path', 'lch', 'wup']:
                    scores[score_name].append(self.verb_similarity[verb_pair][score_name])
                return {'path': statistics.mean(scores['path']),
                        'lch': statistics.mean(scores['lch']),
                        'wup': statistics.mean(scores['wup'])}
            else:
                synset_pairs = itertools.product(list(pair[0].values())[0], list(pair[1].values())[0])
                for synset_pair in synset_pairs:
                    for score_name, score_function in scores_functions.items():
                        score = score_function(synset_pair[0], synset_pair[1])
                        scores[score_name].append(score)
                # updating the dict
                result = {'path': statistics.mean(scores['path']) if len(scores['path']) > 0 else 0,
                          'lch': statistics.mean(scores['lch']) if len(scores['lch']) > 0 else 0,
                          'wup': statistics.mean(scores['wup']) if len(scores['wup']) > 0 else 0}
                self.verb_similarity[verb_pair] = result
                return result

        def local_wn_cosine(synsets):
            """
            :param synsets:
            :return:
            """
            similarity_scores = {'path': list(), 'lch': list(), 'wup': list()}
            if len(synsets) <= 1:
                return {'path': 0, 'lch': 0, 'wup': 0}
            else:
                j = 0
                while j + 1 < len(synsets):
                    result = synset_pair_similarity((synsets[j], synsets[j + 1]))
                    for score_name in ['path', 'lch', 'wup']:
                        similarity_scores[score_name].append(result[score_name])
                    j += 1
                return {'path': statistics.mean(similarity_scores['path']) if len(similarity_scores['path']) > 0 else 0,
                        'lch': statistics.mean(similarity_scores['lch']) if len(similarity_scores['lch']) > 0 else 0,
                        'wup': statistics.mean(similarity_scores['wup']) if len(similarity_scores['wup']) > 0 else 0}

        def global_wn_overlap(pairs):
            """
            computing the wordnet verb overlap among pairs
            :param pairs: list of pair items where each pair has two elements where each element is a list of synsets
            :return:
            """
            similarity_scores = {'path': list(), 'lch': list(), 'wup': list()}
            for pair in pairs:
                result = synset_pair_similarity(pair)
                for score_name in result.keys():
                    similarity_scores[score_name].append(result[score_name])

            return {'path': statistics.mean(similarity_scores['path']) if len(similarity_scores['path']) > 0 else 0,
                    'lch': statistics.mean(similarity_scores['lch']) if len(similarity_scores['lch']) > 0 else 0,
                    'wup': statistics.mean(similarity_scores['wup']) if len(similarity_scores['wup']) > 0 else 0}

        token_synsets = dict()
        for p_s_id, token_ids in token_ids_by_sentence.items():
            p_id = p_s_id.split('_')[0]
            if p_id not in token_synsets:
                token_synsets[p_id] = list()
            current_synsets = list()
            for u_id in token_ids:
                row = df_doc.loc[df_doc['u_id'] == u_id]
                if row.iloc[0]['token_pos'] in pos_tags:
                    token = row.iloc[0]['token_text']
                    synsets = set(wn.synsets(token, wn.VERB))
                    if len(synsets) > 0:
                        current_synsets.append({token: synsets})
            if len(current_synsets) > 0:
                token_synsets[p_id].append(current_synsets)

        scores_1p = list()
        scores_ap = list()

        synsets_flat = list()

        for p_id, synsets_by_sentences in token_synsets.items():
            # *** consecutive (local) cosine ***
            if len(synsets_by_sentences) <= 1:
                scores_1p.append({'path': 0, 'lch': 0, 'wup': 0})
            else:
                i = 0
                while i + 1 < len(synsets_by_sentences):
                    pairs = list(itertools.product(synsets_by_sentences[i], synsets_by_sentences[i + 1]))
                    scores_1p.append(global_wn_overlap(pairs))
                    i += 1

            # *** global cosine ***
            t_synsets = list()  # all synsets of all tokens in one paragraph
            for item in synsets_by_sentences:
                t_synsets.extend(item)
                synsets_flat.extend(item)
            all_pairs = itertools.combinations(t_synsets, r=2)
            scores_ap.append(global_wn_overlap(all_pairs))

        SMCAUSwn_1p_path = statistics.mean([item['path'] for item in scores_1p])
        SMCAUSwn_1p_lch = statistics.mean([item['lch'] for item in scores_1p])
        SMCAUSwn_1p_wup = statistics.mean([item['wup'] for item in scores_1p])
        SMCAUSwn_ap_path = statistics.mean([item['path'] for item in scores_ap])
        SMCAUSwn_ap_lch = statistics.mean([item['lch'] for item in scores_ap])
        SMCAUSwn_ap_wup = statistics.mean([item['wup'] for item in scores_ap])

        # computing global and local indices ignoring the paragraphs
        all_pairs = itertools.combinations(synsets_flat, r=2)
        SMCAUSwn_a = global_wn_overlap(all_pairs)
        SMCAUSwn_1 = local_wn_cosine(synsets_flat)

        SMCAUSwn_1_path = SMCAUSwn_1['path']
        SMCAUSwn_1_lch = SMCAUSwn_1['lch']
        SMCAUSwn_1_wup = SMCAUSwn_1['wup']
        SMCAUSwn_a_path = SMCAUSwn_a['path']
        SMCAUSwn_a_lch = SMCAUSwn_a['lch']
        SMCAUSwn_a_wup = SMCAUSwn_a['wup']

        return {'SMCAUSwn_1p_path': SMCAUSwn_1p_path,
                'SMCAUSwn_1p_lch': SMCAUSwn_1p_lch,
                'SMCAUSwn_1p_wup': SMCAUSwn_1p_wup,
                'SMCAUSwn_ap_path': SMCAUSwn_ap_path,
                'SMCAUSwn_ap_lch': SMCAUSwn_ap_lch,
                'SMCAUSwn_ap_wup': SMCAUSwn_ap_wup,
                'SMCAUSwn_1_path': SMCAUSwn_1_path,
                'SMCAUSwn_1_lch': SMCAUSwn_1_lch,
                'SMCAUSwn_1_wup': SMCAUSwn_1_wup,
                'SMCAUSwn_a_path': SMCAUSwn_a_path,
                'SMCAUSwn_a_lch': SMCAUSwn_a_lch,
                'SMCAUSwn_a_wup': SMCAUSwn_a_wup}

    def _compute_PCREF(self, sentence_embeddings):
        """
        Computing Text Easability PC Referential cohesion
        :param sentence_embeddings: embeddings of sentences in a document
        :return:
        """

        # local: only consecutive sentences either in a paragraph or entire text
        # global: all sentence pairs in a paragraph or entire text

        all_embeddings = list()

        # flattening the embedding list
        for p_id, embeddings in sentence_embeddings.items():
            for embedding in embeddings:
                all_embeddings.append(embedding)

        local_cosine = self._local_cosine(all_embeddings)
        global_cosine = self._global_cosine(all_embeddings)

        del all_embeddings

        local_scores = dict()
        global_scores = dict()
        for p_id, embeddings in sentence_embeddings.items():
            local_scores[p_id] = self._local_cosine(embeddings)
            global_scores[p_id] = self._global_cosine(embeddings)

        # *_p means computed at paragraph-level in contrast to the first case where we ignored paragraphs
        local_cosine_p = statistics.mean([local_scores[p_id] for p_id in local_scores.keys()])
        global_cosine_p = statistics.mean([global_scores[p_id] for p_id in global_scores.keys()])

        return local_cosine, global_cosine, local_cosine_p, global_cosine_p


class GIS:
    def __init__(self):
        self.wolfe_mean_sd = {'SMCAUSlsa': {'mean': 0.097, 'sd': 0.04},
                              'SMCAUSwn': {'mean': 0.553, 'sd': 0.096},
                              'WRDIMGc': {'mean': 410.346, 'sd': 24.994},
                              'WRDHYPnv': {'mean': 1.843, 'sd': 0.26}}
        self.gispy_columns = ["CoREF", "PCREF1", "PCREFa", "PCREF1p", "PCREFap", "PCDC", "SMCAUSe_1", "SMCAUSe_a",
                              "SMCAUSe_1p", "SMCAUSe_ap", "SMCAUSwn", "PCCNC", "WRDIMGc", "WRDHYPnv"]
        self.cohmetrix_columns = ["SMCAUSlsa", "SMCAUSwn", "WRDIMGc", "WRDHYPnv"]

    def _z_score(self, df, index_name, wolfe=False):
        if wolfe:
            return df[index_name].map(
                lambda x: (x - self.wolfe_mean_sd[index_name]['mean']) / self.wolfe_mean_sd[index_name]['sd'])
        else:
            return stats.zscore(df[index_name], nan_policy='omit')

    def score(self, df, variables: dict, wolfe=False, gispy=False):
        """
        computing Gist Inference Score (GIS) based on the following paper:
        https://link.springer.com/article/10.3758/s13428-019-01284-4
        use this method when values of indices are computed using CohMetrix
        :param df: a dataframe that contains coh-metrix indices
        :param variables: a dictionary of information of variables we need to compute the GIS score
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

        columns = self.gispy_columns if gispy else self.cohmetrix_columns
        for column in columns:
            df["z{}".format(column)] = self._z_score(df, index_name=column, wolfe=wolfe)

        # computing the Gist Inference Score (GIS)
        for idx, row in df.iterrows():
            gis = 0
            for variable_name, variable in variables.items():
                gis += variable['sign'] * statistics.mean([row[index_name] for index_name in variable['vars']]) * \
                       variable['flag']
            df.loc[idx, "gis"] = gis

        return df
