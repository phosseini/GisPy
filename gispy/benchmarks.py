import json
import statistics

import pandas
import pandas as pd
from scipy import stats as st

from gist import GIS
from utils import read_word_text


class Wolfe:
    def __init__(self):
        self.text_files_folder_path = '../data/documents'

    def convert_opinion_report_data(self, folder_path, create_text_file=False):
        df = pd.DataFrame(columns=['title', 'type', 'url', 'text'])
        text = read_word_text(folder_path)
        lines = text.split('\n')
        lines = [line for line in lines if line.strip() != '']
        i = 2
        doc_text = list()
        doc_title = lines[0]
        doc_url = lines[1]

        while i < len(lines):
            if lines[i].startswith('Opinion-') or lines[i].startswith('Report-'):
                df = df.append({'title': doc_title, 'type': doc_title.split('-')[0].lower(), 'url': doc_url,
                                'text': '\n'.join(doc_text)}, ignore_index=True)
                doc_title = lines[i]
                doc_url = lines[i + 1]
                doc_text = list()
                i += 2
            else:
                doc_text.append(lines[i])
                i += 1

        # saving the last article
        df = df.append(
            {'title': doc_title, 'type': doc_title.split('-')[0].lower(), 'url': doc_url, 'text': '\n'.join(doc_text)},
            ignore_index=True)

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
        lines = [line for line in lines if line.strip() != '']

        # ARTICLE
        # METHODS
        # DISCUSSION

        methods = list()
        discussion = list()

        i = 0

        while i < len(lines):
            if lines[i].strip().lower().startswith('article'):
                article_title = ' '.join(lines[i].split(':')[1:]).strip()
                i += 2
                while i < len(lines) and not lines[i].lower().startswith('discussion'):
                    methods.append(lines[i])
                    i += 1
                i += 1
                while i < len(lines) and not lines[i].strip().lower().startswith('article'):
                    discussion.append(lines[i])
                    i += 1

                if article_title != "" and len(methods) > 0 and len(discussion) > 0:
                    df = df.append(
                        {'title': article_title, 'methods': '\n'.join(methods), 'discussion': '\n'.join(discussion)},
                        ignore_index=True)
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

    @staticmethod
    def wolfe_eval(input_file, gist_prefix, no_gist_prefix, variables, use_wolfe_vars=False, use_gispy_vars=False):
        df = pd.read_csv(input_file)
        df = GIS().score(df, variables, wolfe=use_wolfe_vars, gispy=use_gispy_vars)

        id_col = 'd_id'

        df_yes = df[df[id_col].apply(lambda x: x.split("_")[0].startswith(gist_prefix))]
        df_no = df[df[id_col].apply(lambda x: x.split("_")[0].startswith(no_gist_prefix))]

        ttest_result = st.ttest_ind(a=list(df_yes['gis']), b=list(df_no['gis']), equal_var=True)

        results = {'mean_gist_yes': statistics.mean(list(df_yes['gis'])),
                   'mean_gist_no': statistics.mean(list(df_no['gis'])),
                   'ttest_statistic': ttest_result.statistic,
                   'ttest_pvalue': ttest_result.pvalue}

        return results, df_yes, df_no


class SummEval:
    def __init__(self):
        pass

    @staticmethod
    def convert_summeval(input_file_path):
        with open(input_file_path, 'r') as input_file:
            lines = input_file.readlines()

        df = pd.DataFrame(columns=["d_id", "text",
                                   "coherence_expert",
                                   "coherence_turker",
                                   "consistency_expert",
                                   "consistency_turker",
                                   "fluency_expert",
                                   "fluency_turker",
                                   "relevance_expert",
                                   "relevance_turker"
                                   ])

        for line in lines:
            line = json.loads(line)

            fields_expert = {'coherence': [], 'consistency': [], 'fluency': [], 'relevance': []}
            fields_turker = {'coherence': [], 'consistency': [], 'fluency': [], 'relevance': []}

            anns_expert = line['expert_annotations']
            anns_turker = line['turker_annotations']

            for field in fields_expert.keys():
                fields_expert[field] = [ann[field] for ann in anns_expert]
                fields_turker[field] = [ann[field] for ann in anns_turker]

            for key, value in fields_expert.items():
                fields_expert[key] = sum(value) / len(value)
            for key, value in fields_turker.items():
                fields_turker[key] = sum(value) / len(value)

            df = df.append({"d_id": line["id"], "text": line['decoded'],
                            'coherence_expert': fields_expert['coherence'],
                            'coherence_turker': fields_turker['coherence'],
                            'consistency_expert': fields_expert['consistency'],
                            'consistency_turker': fields_turker['consistency'],
                            'fluency_expert': fields_expert['fluency'],
                            'fluency_turker': fields_turker['fluency'],
                            'relevance_expert': fields_expert['relevance'],
                            'relevance_turker': fields_turker['relevance'], },
                           ignore_index=True)
        df.reset_index()
        return df

    @staticmethod
    def summeval_eval(input_file, variables, use_wolfe_vars=False, use_gispy_vars=False):
        df = pd.read_csv(input_file)
        df = GIS().score(df, variables, wolfe=use_wolfe_vars, gispy=use_gispy_vars)

        df_og = pandas.read_excel('../data/SummEval.xlsx')
        labels = ['coherence_expert', 'coherence_turker']
        scores = dict()
        for label in labels:
            scores[label] = dict()
            for i in range(1, 6):
                scores[label][i] = list()

        for idx, row in df_og.iterrows():
            record = df[df['d_id'].apply(lambda x: str(x.strip('.txt')) == 'se_{}_{}'.format(idx, row['id']))]
            if len(record) == 1:
                gis = float(record.iloc[0]['gis'])

            for label_name in scores.keys():
                label = float(row[label_name])
                for i in range(1, 6):
                    if i <= label < i + 1:
                        scores[label_name][i].append(gis)

        return scores
