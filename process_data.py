import pandas as pd
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='data/ec_master_for_hong.csv', help='input csv data')
    parser.add_argument('--output_data', type=str, default='data/ec_master.csv', help='processed data')

    args = parser.parse_args()
    input_data = args.input_data
    output_data = args.output_data
    print(os.getcwd())
    input_df = pd.read_csv(input_data)
    print(f'before removing all null columns: {input_df.shape}')
    input_df = input_df.dropna(axis=1, how='all')
    print(f'after removing all null columns: {input_df.shape}')
    pd.set_option('display.max_rows', None)
    print(input_df.isnull().sum()/input_df.shape[0])
    input_df.to_csv(output_data, index=False)
    basename = os.path.splitext(output_data)[0]
    input_df_not_null = input_df[input_df['Y/N/P'].notnull()]
    input_df_null = input_df[input_df['Y/N/P'].isna()]
    input_df_not_null.to_csv(f'{basename}_not_null.csv', index=False)
    input_df_null.to_csv(f'{basename}_null.csv', index=False)
