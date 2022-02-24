import re

import spacy
import pandas as pd

from os import listdir, path
from os.path import isfile, join

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('sentencizer')


class DataReader:
    def __init__(self):
        self.input_path = "../data/input/"

    def list_input_files(self):
        """
        listing all the input files
        :return:
        """
        all_files = [f for f in listdir(self.input_path) if isfile(join(self.input_path, f))]
        txt_files = []
        for file in all_files:
            if file.endswith(".txt") and not file.startswith("~$"):
                txt_files.append(self.input_path + file)
        return txt_files

    def load_input_files(self, count=5):
        """
        reading input documents
        :param count:
        :return: a string with documents separated by '\n\n'
        """
        files = self.list_input_files()
        n = 0
        docs = ""
        for file in files:
            with open(file, "r") as f:
                docs += f.read() + "\n\n"
            if n >= count:
                return docs
            n += 1
        return docs


def convert_doc(doc_text):
    """
    converting a document to tokens with meta-information (e.g. POS tags, vector embeddings)
    :param doc_text: text of a document
    :return:
    """

    def normalize_text(input_text):
        input_text = re.sub(r'\n+', '\n', input_text).strip()
        return input_text

    # u_id: unique identifier
    df_doc = pd.DataFrame(columns=["u_id", "p_id", "s_id", "token_id", "token_text", "token_lemma", "token_pos"])
    token_embeddings = dict()
    doc_text = normalize_text(doc_text)
    paragraphs = doc_text.split('\n')
    p_id = 0
    u_id = 0
    for paragraph in nlp.pipe(paragraphs, disable=["parser"]):
        s_id = 0
        for sent in paragraph.sents:
            tokens = [t for t in sent]
            t_id = 0
            for token in tokens:
                df_doc = df_doc.append({"u_id": u_id,
                                        "p_id": p_id,
                                        "s_id": s_id,
                                        "token_id": t_id,
                                        "token_text": token.text.strip(),
                                        "token_lemma": token.lemma_.strip(),
                                        "token_pos": token.pos_},
                                       ignore_index=True)
                token_embeddings[u_id] = token.tensor
                u_id += 1
                t_id += 1
            s_id += 1
        p_id += 1

    return df_doc, token_embeddings
