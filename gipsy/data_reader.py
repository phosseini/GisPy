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


def convert_docs(docs_string):
    """
    converting docs tokens
    :param docs_string: a string in which documents are separated by '\n\n' and paragraphs by '\n'
    :return:
    """

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    df_docs = pd.DataFrame(columns=["d_id", "p_id", "sen_id", "token_id", "token_text", "token_lemma", "token_pos"])

    # documents and paragraphs should be separated by '\n\n' and '\n', respectively.
    documents = docs_string.split("\n\n")
    d_id = 0
    for doc in documents:
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
                                              "sen_id": s_id,
                                              "token_id": t_id,
                                              "token_text": token.string.strip(),
                                              "token_lemma": token.lemma_.strip(),
                                              "token_pos": token.pos_},
                                             ignore_index=True)
                    t_id += 1
                s_id += 1
            p_id += 1
        d_id += 1

    return df_docs
