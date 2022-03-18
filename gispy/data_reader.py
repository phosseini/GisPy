import re

import spacy
from spacy.language import Language
import pandas as pd
import numpy as np

from os import listdir, path
from os.path import isfile, join


@Language.factory('tensor2attr')
class Tensor2Attr:
    """
    source code: https://applied-language-technology.mooc.fi/html/notebooks/part_iii/05_embeddings_continued.html
    """

    # We continue by defining the first method of the class,
    # __init__(), which is called when this class is used for
    # creating a Python object. Custom components in spaCy
    # require passing two variables to the __init__() method:
    # 'name' and 'nlp'. The variable 'self' refers to any
    # object created using this class!
    def __init__(self, name, nlp):
        # We do not really do anything with this class, so we
        # simply move on using 'pass' when the object is created.
        pass

    # The __call__() method is called whenever some other object
    # is passed to an object representing this class. Since we know
    # that the class is a part of the spaCy pipeline, we already know
    # that it will receive Doc objects from the preceding layers.
    # We use the variable 'doc' to refer to any object received.
    def __call__(self, doc):
        # When an object is received, the class will instantly pass
        # the object forward to the 'add_attributes' method. The
        # reference to self informs Python that the method belongs
        # to this class.
        self.add_attributes(doc)

        # After the 'add_attributes' method finishes, the __call__
        # method returns the object.
        return doc

    # Next, we define the 'add_attributes' method that will modify
    # the incoming Doc object by calling a series of methods.
    def add_attributes(self, doc):
        # spaCy Doc objects have an attribute named 'user_hooks',
        # which allows customising the default attributes of a
        # Doc object, such as 'vector'. We use the 'user_hooks'
        # attribute to replace the attribute 'vector' with the
        # Transformer output, which is retrieved using the
        # 'doc_tensor' method defined below.
        doc.user_hooks['vector'] = self.doc_tensor

        # We then perform the same for both Spans and Tokens that
        # are contained within the Doc object.
        doc.user_span_hooks['vector'] = self.span_tensor
        doc.user_token_hooks['vector'] = self.token_tensor

        # We also replace the 'similarity' method, because the
        # default 'similarity' method looks at the default 'vector'
        # attribute, which is empty! We must first replace the
        # vectors using the 'user_hooks' attribute.
        doc.user_hooks['similarity'] = self.get_similarity
        doc.user_span_hooks['similarity'] = self.get_similarity
        doc.user_token_hooks['similarity'] = self.get_similarity

    # Define a method that takes a Doc object as input and returns
    # Transformer output for the entire Doc.
    def doc_tensor(self, doc):
        # Return Transformer output for the entire Doc. As noted
        # above, this is the last item under the attribute 'tensor'.
        # Average the output along axis 0 to handle batched outputs.
        return doc._.trf_data.tensors[-1].mean(axis=0)

    # Define a method that takes a Span as input and returns the Transformer
    # output.
    def span_tensor(self, span):
        # Get alignment information for Span. This is achieved by using
        # the 'doc' attribute of Span that refers to the Doc that contains
        # this Span. We then use the 'start' and 'end' attributes of a Span
        # to retrieve the alignment information. Finally, we flatten the
        # resulting array to use it for indexing.
        tensor_ix = span.doc._.trf_data.align[span.start: span.end].data.flatten()

        # Fetch Transformer output shape from the final dimension of the output.
        # We do this here to maintain compatibility with different Transformers,
        # which may output tensors of different shape.
        out_dim = span.doc._.trf_data.tensors[0].shape[-1]

        # Get Token tensors under tensors[0]. Reshape batched outputs so that
        # each "row" in the matrix corresponds to a single token. This is needed
        # for matching alignment information under 'tensor_ix' to the Transformer
        # output.
        tensor = span.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]

        # Average vectors along axis 0 ("columns"). This yields a 768-dimensional
        # vector for each spaCy Span.
        return tensor.mean(axis=0)

    # Define a function that takes a Token as input and returns the Transformer
    # output.
    def token_tensor(self, token):
        # Get alignment information for Token; flatten array for indexing.
        # Again, we use the 'doc' attribute of a Token to get the parent Doc,
        # which contains the Transformer output.
        tensor_ix = token.doc._.trf_data.align[token.i].data.flatten()

        # Fetch Transformer output shape from the final dimension of the output.
        # We do this here to maintain compatibility with different Transformers,
        # which may output tensors of different shape.
        out_dim = token.doc._.trf_data.tensors[0].shape[-1]

        # Get Token tensors under tensors[0]. Reshape batched outputs so that
        # each "row" in the matrix corresponds to a single token. This is needed
        # for matching alignment information under 'tensor_ix' to the Transformer
        # output.
        tensor = token.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]

        # Average vectors along axis 0 (columns). This yields a 768-dimensional
        # vector for each spaCy Token.
        return tensor.mean(axis=0)

    # Define a function for calculating cosine similarity between vectors
    def get_similarity(self, doc1, doc2):
        # Calculate and return cosine similarity
        return np.dot(doc1.vector, doc2.vector) / (doc1.vector_norm * doc2.vector_norm)


nlp_trf = spacy.load('en_core_web_trf')
nlp_trf.add_pipe('tensor2attr')
nlp_trf.add_pipe('sentencizer')


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

    # u_id: unique identifier
    df_doc = pd.DataFrame(columns=["u_id", "p_id", "s_id", "token_id", "token_text", "token_lemma", "token_pos"])
    token_embeddings = dict()
    paragraphs = doc_text.split('\n')
    p_id = 0
    u_id = 0
    for paragraph in nlp_trf.pipe(paragraphs, disable=["parser"]):
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
                token_embeddings[u_id] = token.vector
                u_id += 1
                t_id += 1
            s_id += 1
        p_id += 1

    return df_doc, token_embeddings


class GisPyData:
    def __init__(self):
        pass

    @staticmethod
    def get_gispy_index_columns():
        return ["DESPC", "DESSC", "CoREF", "PCREF1", "PCREFa", "PCREF1p", "PCREFap", "PCDC", "SMCAUSe_1",
                "SMCAUSe_a", "SMCAUSe_1p", "SMCAUSe_ap",
                "SMCAUSwn_1p_path", "SMCAUSwn_1p_lch", "SMCAUSwn_1p_wup",
                "SMCAUSwn_ap_path", "SMCAUSwn_ap_lch", "SMCAUSwn_ap_wup",
                "SMCAUSwn_1_path", "SMCAUSwn_1_lch", "SMCAUSwn_1_wup",
                "SMCAUSwn_a_path", "SMCAUSwn_a_lch", "SMCAUSwn_a_wup",
                "PCCNC", "WRDIMGc", "WRDHYPnv"]
