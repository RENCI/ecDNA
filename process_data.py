############################################
# This script read the initial data and remove all null columns
# and all rows with Y/N/P target column being null to output
# processed data to a file.
############################################
import pandas as pd
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='data/df_CCLE_Mitelman_for_ML.csv', help='input csv data')
    parser.add_argument('--output_data', type=str, default='data/CCLE_Mitelman_for_ML.csv', help='processed data')
    parser.add_argument('--classification_column', type=str, default='ECDNA_classification',
                        help='the classification column; for initial data, it is Y/N/P, for updated data, it is '
                             'ECDNA_classification')

    args = parser.parse_args()
    input_data = args.input_data
    output_data = args.output_data
    classification_column = args.classification_column

    print(os.getcwd())
    input_df = pd.read_csv(input_data)
    if 'Unnamed: 0' in input_df.columns:
        input_df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
    if 'XY_fixed' in input_df.columns and 'XY_chromosomes' in input_df.columns:
        input_df.drop(columns=['XY_chromosomes'], inplace=True)
        input_df.rename(columns={'XY_fixed': 'XY_chromosomes'}, inplace=True)
    print(f'before removing all null columns: {input_df.shape}')
    input_df = input_df.dropna(axis=1, how='all')
    print(f'after removing all null columns: {input_df.shape}')
    pd.set_option('display.max_rows', None)
    # pd.set_option('max_seq_items', None)
    basename = os.path.splitext(output_data)[0]
    input_df_null = input_df[input_df[classification_column].isna()]
    if not input_df_null.empty:
        input_df_null.to_csv(f'{basename}_null.csv', index=False)
        # remove rows with target column Y/N/P being null
        input_df = input_df[input_df[classification_column].notnull()]
        print(f'after removing all rows with {classification_column} targe column being null: {input_df.shape}')
    print((input_df.isnull().sum()/input_df.shape[0]).sort_values())
    input_df.to_csv(output_data, index=False)
