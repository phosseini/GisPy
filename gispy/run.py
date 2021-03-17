from gist import GIST
from data_reader import convert_docs
from data_reader import DataReader

# loading documents
dr = DataReader()
doc = dr.load_input_files(count=1)

# converting document to data frame
doc_df = convert_docs(doc)

# computing gist
gist = GIST(doc=doc_df)
print(gist.compute_SMCAUSwn())
