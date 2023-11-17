import pandas as pd
import matplotlib.pyplot as plt
import argparse
from impute_features import columns_with_nulls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='data/ec_master_simplified.csv', help='input csv data')

    args = parser.parse_args()
    input_data = args.input_data

    input_df = pd.read_csv(input_data)

    input_df[columns_with_nulls].hist(bins=50, figsize=(40, 30))
    plt.show()
