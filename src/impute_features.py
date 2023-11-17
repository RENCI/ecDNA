############################################
# This script should be run after running convert_data_to_numerical.py.
# It imputes all columns with missing data to get data ready for
# training a classifer model, then output imputed data to a file.
############################################

import pandas as pd
import argparse

ploidy_cols = [
    'modal_range_numeric',
    'modal chromosome number',
    'marker chromosomes (average #)'
]

# map definition to (range, mean)
ploidy_definition_map = {
    0: (6, 85), # '82-88',
    3: (13, 65), # '58-71',
    2: (0, 69), # '69',
    1: (4, 78), # '76-80',
    -1: (-1, -1),
    6: (0, 46), # '46',
    5: (30, 61), # '46-76',
    7: (1, 44), # '44–45',
    4: (10, 29) # '24-34'
}


def impute_ploidy_columns(row, col):
    if pd.isnull(row[col]):
        if 'range' in col:
            return ploidy_definition_map[row['ploidy_classification']][0]
        else:
            return ploidy_definition_map[row['ploidy_classification']][1]
    else:
        return row[col]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='data/CCLE_Mitelman_for_ML_numerical.csv', help='input csv data')
    parser.add_argument('--output_data', type=str, default='data/CCLE_Mitelman_for_ML_imputed.csv', help='imputed data')

    args = parser.parse_args()
    input_data = args.input_data
    output_data = args.output_data
    input_df = pd.read_csv(input_data)


    for col in ploidy_cols:
        input_df[col] = input_df.apply(lambda row: impute_ploidy_columns(row, col), axis=1)

    input_df.to_csv(output_data, index=False)

