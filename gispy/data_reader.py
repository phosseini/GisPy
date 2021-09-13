import re

import spacy
import pandas as pd

from os import listdir, path
from os.path import isfile, join


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


def convert_docs(df_raw_docs):
    """
    converting string documents to tokens with meta-information (e.g. POS tags)
    :param df_raw_docs: a data frame with two columns: ['d_id', 'text']
    :return:
    """

    def normalize_text(input_text):
        input_text = re.sub(r'\n+', '\n', input_text).strip()
        return input_text

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('sentencizer')

    df_docs = pd.DataFrame(columns=["d_id", "p_id", "s_id", "token_id", "token_text", "token_lemma", "token_pos"])

    for idx, row in df_raw_docs.iterrows():
        d_id = row['d_id']
        doc = normalize_text(row['text'])
        paragraphs = doc.split('\n')
        p_id = 0
        for paragraph in nlp.pipe(paragraphs, disable=["parser"]):
            s_id = 0
            for sent in paragraph.sents:
                tokens = [t for t in sent]
                t_id = 0
                for token in tokens:
                    df_docs = df_docs.append({"d_id": d_id,
                                              "p_id": p_id,
                                              "s_id": s_id,
                                              "token_id": t_id,
                                              "token_text": token.text.strip(),
                                              "token_lemma": token.lemma_.strip(),
                                              "token_pos": token.pos_},
                                             ignore_index=True)
                    t_id += 1
                s_id += 1
            p_id += 1

    return df_docs
