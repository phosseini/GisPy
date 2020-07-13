from nltk.corpus import wordnet as wn


class GIST:
    def __init__(self, doc):
        self.doc = doc

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
