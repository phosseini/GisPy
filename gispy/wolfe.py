import statistics
import pandas as pd
from scipy import stats as st

from gist import GIS
from utils import read_word_text


class WolfeData:
    def __init__(self):
        self.text_files_folder_path = '../data/documents'

    def convert_opinion_report_data(self, folder_path, create_text_file=False):
        df = pd.DataFrame(columns=['title', 'type', 'url', 'text'])
        text = read_word_text(folder_path)
        lines = text.split('\n')

        i = 4
        doc_text = list()
        doc_title = lines[0]
        doc_url = lines[2]

        while i < len(lines):
            if lines[i].startswith('Opinion-') or lines[i].startswith('Report-'):
                df = df.append({'title': doc_title, 'type': doc_title.split('-')[0].lower(), 'url': doc_url,
                                'text': '\n'.join(doc_text)}, ignore_index=True)
                doc_title = lines[i]
                doc_url = lines[i + 2]
                doc_text = list()
                i += 4
            else:
                doc_text.append(lines[i])
                i += 1

        df.reset_index()
        if create_text_file:
            for idx, row in df.iterrows():
                with open('{}/{}_{}.txt'.format(self.text_files_folder_path, row['type'], idx), 'w') as text_file:
                    text_file.write(row['text'])
        return df

    def convert_methods_discussion_data(self, folder_path, create_text_file=False):
        df = pd.DataFrame(columns=['title', 'methods', 'discussion'])
        text = read_word_text(folder_path)
        lines = text.split('\n')

        # ARTICLE
        # METHODS
        # DISCUSSION

        article_title = str()
        methods = list()
        discussion = list()

        i = 0

        while i < len(lines):
            if lines[i].strip().lower().startswith('article'):
                article_title = ' '.join(lines[i].split(':')[1:]).strip()
                i += 4
                while i < len(lines) and not lines[i].lower().startswith('discussion'):
                    methods.append(lines[i])
                    i += 1
                i += 2
                while i < len(lines) and not lines[i].strip().lower().startswith('article'):
                    discussion.append(lines[i])
                    i += 1

                if article_title != "" and len(methods) > 0 and len(discussion) > 0:
                    df = df.append(
                        {'title': article_title, 'methods': '\n'.join(methods), 'discussion': '\n'.join(discussion)},
                        ignore_index=True)
                    article_title = str()
                    methods = list()
                    discussion = list()

        df.reset_index()
        if create_text_file:
            for idx, row in df.iterrows():
                with open('{}/methods_{}.txt'.format(self.text_files_folder_path, idx), 'w') as text_file:
                    text_file.write(row['methods'])
                with open('{}/discussion_{}.txt'.format(self.text_files_folder_path, idx), 'w') as text_file:
                    text_file.write(row['discussion'])
        return df


def wolfe_eval(input_file, gist_prefix, no_gist_prefix, use_wolfe_vars=False, use_gispy_vars=False):
    df = pd.read_csv(input_file)
    df = GIS().score(df, wolfe=use_wolfe_vars, gispy=use_gispy_vars)

    id_col = 'd_id'

    df_yes = df[df[id_col].apply(lambda x: x.split("_")[0].startswith(gist_prefix))]
    df_no = df[df[id_col].apply(lambda x: x.split("_")[0].startswith(no_gist_prefix))]

    ttest_result = st.ttest_ind(a=list(df_yes['gis']), b=list(df_no['gis']), equal_var=True)

    results = {'mean_gist_yes': statistics.mean(list(df_yes['gis'])),
               'mean_gist_no': statistics.mean(list(df_no['gis'])),
               'ttest_statistic': ttest_result.statistic,
               'ttest_pvalue': ttest_result.pvalue}

    return results, df_yes, df_no
