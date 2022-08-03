import sys
import json
import timeit
import os.path
from gist import GIST, GIS
from data_reader import GisPyData


def main(argv):
    usage_msg = 'Usage: python run.py [OUTPUT_FILE_NAME]'

    if not os.path.exists('gispy_config.json'):
        print('Error: gist_config.json could not be found. please put the config file in current directory.')
        exit(1)

    if not os.path.exists('gis_config.json'):
        print('Error: gis_config.json could not be found. The config file is required for computing GIS.')
        exit(1)
    else:
        with open('gis_config.json') as f_config:
            gis_config = json.load(f_config)
            if any(var not in list(GisPyData().get_gispy_index_columns()) for var in list(gis_config.keys())):
                print("Some variables in the gis_config.json are not among the GisPy indices. Check variables' names")
                exit(1)

    if len(argv) >= 2:
        if argv[-1].endswith('.csv'):
            file_path = argv[-1]
        else:
            print('error: OUTPUT_FILE_NAME should be *.csv')
            exit(1)

        start = timeit.default_timer()
        df_scores = GIST(docs_path='../data/documents').compute_indices()

        try:
            vars_dict = GisPyData().convert_config_to_vars_dict(gis_config)
            df_scores = GIS().score(df_scores, vars_dict, gispy=True)
        except Exception as e:
            print('Error in computing the GIS.')
            print(e)

        stop = timeit.default_timer()

        # saving the result
        df_scores.to_csv(file_path)
        print('Results are saved at: /{}'.format(file_path))
        print('Running time: {}\n'.format(stop - start))
    else:
        print(usage_msg)
        exit(1)


if __name__ == '__main__':
    main(sys.argv)
