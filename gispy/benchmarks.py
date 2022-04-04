import json
import statistics

import pandas
import pandas as pd
from scipy import stats as st
import matplotlib.pyplot as plt

from gist import GIS
from data_reader import GisPyData
from utils import read_word_text


class Wolfe:
    def __init__(self):
        self.text_files_folder_path = '../data/documents'

    def convert_reports_editorials_data(self, folder_path, create_text_file=False):
        # Editorials are saved in Wolfe's data as *opinion*, so *Editorials* and *Opinion* are the same category
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
        prefixes = {'report': 'reports', 'opinion': 'editorials'}
        if create_text_file:
            for idx, row in df.iterrows():
                with open('{}/{}_{}.txt'.format(self.text_files_folder_path, prefixes[row['type']], idx),
                          'w') as text_file:
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
    def wolfe_eval(input_file, prefix_names, variables, doc_filter, use_wolfe_vars=False, use_gispy_vars=False):
        df = pd.read_csv(input_file)
        df = GIS().score(df, variables, wolfe=use_wolfe_vars, gispy=use_gispy_vars)

        id_col = 'd_id'

        df_yes = df[
            df[id_col].apply(lambda x: (len(doc_filter) == 0 or x in doc_filter) and x.split("_")[0].startswith(
                prefix_names['gist_yes']))]
        df_no = df[df[id_col].apply(lambda x: (len(doc_filter) == 0 or x in doc_filter) and x.split("_")[0].startswith(
            prefix_names['gist_no']))]

        ttest_result = st.ttest_ind(a=list(df_yes['gis']), b=list(df_no['gis']), equal_var=True)

        results = {'mean_gist_yes': statistics.mean(list(df_yes['gis'])),
                   'mean_gist_no': statistics.mean(list(df_no['gis'])),
                   'ttest_statistic': ttest_result.statistic,
                   'ttest_pvalue': ttest_result.pvalue}

        return results, df_yes, df_no

    def run_benchmark(self, gispy_indices_file: str, prefix_names: dict, doc_filter=[], use_wolfe_vars=False,
                      use_gispy_vars=True, custom_vars=[], plot=False):
        """
        running GIS score computation on a Wolfe benchmark
        :param gispy_indices_file: the output .csv file generated by gisPy after running it on a collection of documents
        :param prefix_names: dictionary that contains the prefix of different group names in the benchmark
        :param use_wolfe_vars:
        :param use_gispy_vars:
        :param doc_filter: a tuple including only d_ids we want to include in the analysis
        :param custom_vars: a list containing three string values
        :param plot: whether to plot scores
        :return:
        """
        vars_dicts = GisPyData().get_variables_dict(gispy=use_gispy_vars, custom_vars=custom_vars)
        results = pd.DataFrame(
            columns=['vars_name', 'mean_gist_yes', 'mean_gist_no', 'distance', 'ttest_statistic',
                     'ttest_pvalue'])
        plot_id = 0
        for vars_dict in vars_dicts:
            result, df_yes, df_no = Wolfe().wolfe_eval(gispy_indices_file,
                                                       prefix_names,
                                                       vars_dict,
                                                       doc_filter,
                                                       use_wolfe_vars=use_wolfe_vars,
                                                       use_gispy_vars=use_gispy_vars)
            vars_string, vars_list = GisPyData().generate_variables_dict_id(vars_dict)
            vars_string = str(plot_id) + '#' + vars_string
            results = results.append({
                'vars_name': vars_string,
                'mean_gist_yes': result['mean_gist_yes'],
                'mean_gist_no': result['mean_gist_no'],
                'distance': abs(result['mean_gist_yes'] - result['mean_gist_no']),
                'ttest_statistic': result['ttest_statistic'],
                'ttest_pvalue': result['ttest_pvalue']}, ignore_index=True)

            if plot:
                self.plot_scores(df_yes, df_no, vars_list, vars_string)
            plot_id += 1

        results.sort_values('distance', ascending=False, inplace=True)

        return results

    @staticmethod
    def plot_scores(df_gist, df_no_gist, column_names, output_file_name):
        fig, axis = plt.subplots(3, 3, figsize=(7, 7), constrained_layout=True)

        col_idx = 0
        cols = list()
        for i in range(3):
            for j in range(3):
                if col_idx < len(column_names):
                    cols.append([column_names[col_idx], i, j])
                    col_idx += 1

        for col in cols:
            axis[col[1], col[2]].boxplot([df_gist[col[0]], df_no_gist[col[0]]], labels=['Gist', 'No Gist'])
            axis[col[1], col[2]].set_title(col[0])

        axis[2, 1].set_axis_off()
        axis[2, 2].set_axis_off()

        fig.savefig("../data/plots/{}.pdf".format(output_file_name))
        plt.close(fig)


class SummEval:
    def __init__(self):
        pass

    @staticmethod
    def convert_summeval(input_file_path):
        with open(input_file_path, 'r') as input_file:
            lines = input_file.readlines()

        records = pd.DataFrame(columns=["d_id", "text",
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
            record = {"d_id": line["id"], "text": line['decoded']}
            for annotation_type in ['expert_annotations', 'turker_annotations']:
                scores = {'coherence': list(), 'consistency': list(), 'fluency': list(), 'relevance': list()}
                annotations = line[annotation_type]
                # since each text has multiple expert and MTurk annotators, we iterate through all annotations
                for field in scores.keys():
                    scores[field] = [ann[field] for ann in annotations]

                for score_name, score_values in scores.items():
                    record['{}_{}'.format(score_name, annotation_type.split('_')[0])] = statistics.mean(score_values)
            records = records.append(record, ignore_index=True)
        records.reset_index()
        return records

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
