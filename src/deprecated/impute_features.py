############################################
# This script should be run after running extract_features.py.
# It imputes all columns with missing data to get data ready for
# training a classifer model, then output imputed data to a file.
############################################

import pandas as pd
import argparse

ploidy_cols = [
    'modal_range_numeric',
    'modal chromosome number',
    '% polyploidy',
    'marker chromosomes (average #)'
]

impute_with_0_cols = [
    'max_num_frags',
    'max_num_cnv_changes',
    'max_frag_length',
    'num_chr_w_FA',
    'max_cnv'
]

columns_with_nulls = ploidy_cols + impute_with_0_cols

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
    parser.add_argument('--input_data', type=str, default='data/ec_master_simplified.csv', help='input csv data')
    parser.add_argument('--output_data', type=str, default='data/ec_master_imputed.csv', help='imputed data')

    args = parser.parse_args()
    input_data = args.input_data
    output_data = args.output_data
    input_df = pd.read_csv(input_data)

    for col in impute_with_0_cols:
        input_df[col] = input_df[col].fillna(0)

    for col in ploidy_cols:
        input_df[col] = input_df.apply(lambda row: impute_ploidy_columns(row, col), axis=1)

    input_df.to_csv(output_data, index=False)

