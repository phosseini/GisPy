from nltk.corpus import wordnet as wn


class GIST:
    def __init__(self, doc):
        self.doc = doc

    def compute_SMCAUSwn(self):
        """
        computing the WordNet Verb Overlap in a document
        :return:
        """
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
