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
    basename = os.path.splitext(output_data)[0]
    input_df_null = input_df[input_df['Y/N/P'].isna()]
    input_df_null.to_csv(f'{basename}_null.csv', index=False)
    # remove rows with target column Y/N/P being null
    input_df = input_df[input_df['Y/N/P'].notnull()]
    print(f'after removing all rows with Y/N/P targe column being null: {input_df.shape}')
    print((input_df.isnull().sum()/input_df.shape[0]).sort_values())
    input_df.to_csv(output_data, index=False)

