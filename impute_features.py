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

ploidy_definition_map = {
    'hexaploid/octaploid': '138-184',
    'hyperpentaploid': '>115',
    'pseudotetraploid': '92',
    2: '69',
    1: '76-80',
    'hypertetraploid': '>92',
    'near-diploid': '46',
    'near-pseudodiploid': '45-46',
    'polyploid': '>46',
    'hypotetraploid': '76-83',
    'tetraploid': '92',
    6: '46',
    5: '46-76',
    7: '44–45',
    'pseudodiploid': '44/45 or 46/47',
    'near-tetraploid': '81-103',
    'near-triploid': '58-80',
    4: '24-34'
}


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




    input_df.to_csv(output_data, index=False)

