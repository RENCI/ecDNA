import pandas as pd
import matplotlib.pyplot as plt
import argparse


columns_with_nulls = [
    'modal_range_numeric',
    'modal chromosome number',
    '% polyploidy',
    'marker chromosomes (average #)',
    'max_num_frags',
    'max_num_cnv_changes',
    'max_frag_length',
    'num_chr_w_FA',
    'max_cnv'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='data/ec_master_simplified.csv', help='input csv data')

    args = parser.parse_args()
    input_data = args.input_data

    input_df = pd.read_csv(input_data)

    input_df[columns_with_nulls].hist(bins=50, figsize=(40, 30))
    plt.show()
