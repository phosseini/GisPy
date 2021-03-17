from sqlalchemy import create_engine, MetaData, Table, and_
from sqlalchemy.orm import sessionmaker


def find_mrc_word(word, pos):
    """
    finding a word in MRC database
    :param word: word that we're searching in MRC
    :param pos: string part-of-speech (POS) from spaCy tag list. E.g. 'VERB' or 'NOUN'
    :return:
    """
    engine = create_engine('sqlite:///../data/mrc/{name}.db'.format(name='mrc2'))
    table_meta = MetaData(engine)
    table = Table('word', table_meta, autoload=True)
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    tag_map = spacy2mrc_pos()

    # creating POS tag list
    if pos in tag_map:
        tag_list = tag_map[pos]
    else:
        tag_list = ['O']

    records = session.query(table).filter(
        and_(table.columns.word == word.upper(), table.columns.wtype.in_(tag_list))).all()

    return records


def spacy2mrc_pos():
    """
    mapping the spaCy part-of-speech (POS) tags to MRC's WTYPE
    MRC's WTYPE: https://websites.psychology.uwa.edu.au/school/MRCDatabase/mrc2.html#WTYPE
    ** All the POS tags that do not exist in tag_map dictionary can be considered as 'O' in MRC
    :return:
    """

    tag_map = {
        'NOUN': ['N'],
        'PROPN': ['N'],
        'ADJ': ['J'],
        'VERB': ['V', 'P'],
        'ADV': ['A'],
        'ADP': ['A', 'R', 'C'],
        'CCONJ': ['C'],
        'PRON': ['U'],
        'DET': ['U'],
        'INTJ': ['I']
    }

    return tag_map
