############################################
# This script should be run after running convert_data_to_numerical.py.
# It read the all numerical data and drop non-relevant feature columns,
# then output simplified data to a file.
############################################

import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='data/ec_master_numerical.csv', help='input csv data')
    parser.add_argument('--output_data', type=str, default='data/ec_master_simplified.csv',
                        help='data with only relevant features included')

    args = parser.parse_args()
    input_data = args.input_data
    output_data = args.output_data
    input_df = pd.read_csv(input_data)
    # drop all feature columns starting from chr_
    input_df.drop(list(input_df.filter(regex='^chr_\d*_\S')), axis=1, inplace=True)
    # drop all feature columns starting from INV_
    input_df.drop(list(input_df.filter(regex='^INV_\d*')), axis=1, inplace=True)
    # drop all feature columns starting from INS_
    input_df.drop(list(input_df.filter(regex='^INS_\d*')), axis=1, inplace=True)
    # drop all feature columns starting from DEL_ or DUP_ or TRANS_, or ISO_, or DER_
    input_df.drop(list(input_df.filter(regex='^DEL_\d*')), axis=1, inplace=True)
    input_df.drop(list(input_df.filter(regex='^DUP_\d*')), axis=1, inplace=True)
    input_df.drop(list(input_df.filter(regex='^TRANS_\d*')), axis=1, inplace=True)
    input_df.drop(list(input_df.filter(regex='^ISO_\d*')), axis=1, inplace=True)
    input_df.drop(list(input_df.filter(regex='^DER_\d*')), axis=1, inplace=True)
    input_df.to_csv(output_data, index=False)

