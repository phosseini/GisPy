import sys
import timeit
import os.path
from gist import GIST


def main(argv):
    usage_msg = 'Usage: python run.py [OUTPUT_FILE_NAME]'

    if not os.path.exists('gist_config.json'):
        print('error: gist_config.json could not be found. please put the config file in current directory.')
        exit(1)

    if len(argv) >= 2:
        if argv[-1].endswith('.csv'):
            file_path = argv[-1]
        else:
            print('error: OUTPUT_FILE_NAME should be *.csv')
            exit(1)

        start = timeit.default_timer()
        df_scores = GIST(docs_path='../data/documents').compute_scores()
        stop = timeit.default_timer()

        # saving the result
        df_scores.to_csv(file_path)
        print('computing GIS is done. results are saved at /{}'.format(file_path))
        print('Running time: {}'.format(stop - start))
    else:
        print(usage_msg)
        exit(1)


if __name__ == '__main__':
    main(sys.argv)
