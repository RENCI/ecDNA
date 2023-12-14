############################################
# This script should be run after running process_data.py.
# It read the processed data and convert all non-numerical columns
# to numbers, then output converted data to a file.
############################################
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder


ploidy_mapping = {
        'higher ploidies (>=4N)': 0,
        'hypertriploid': 1,
        'triploid': 2,
        'mixed ploidies (<= 3N)': 3,
        'hypotriploid': 4,
        'hyperdiploid': 5,
        'diploid': 6,
        'hypodiploid': 7,
        'aneuploid': -1,
        'polyploid': -1
    }

chromosome_mapping = {
        '-Y': 0,
        'X': 1,
        'Y': 2,
        'XX': 3,
        'XY': 4,
        'derX_or_derY': 5,
        'XX_heterogeneous': 6,
        'XY_heterogeneous': 7,
        'YY': 8,
        'XXX': 9,
        'XXY': 10,
        'XYY': 11,
        'YYY': 12,
        'XYq+': 13,
        'XXYY': 14,
        'XXXX': 15,
        'XXXXX': 16,
        'XXXYY': 17,
        'XXXYYY': 18,
        'XXXXXXX': 19
    }


def map_feature(df_col, mapping):
    # apply custom mapping to map column to convert the column from strings to numbers, -1
    # means unknown
    df_col = df_col.map(mapping)
    df_col = df_col.fillna(-1)
    return df_col.astype('Int8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='data/CCLE_Mitelman_for_ML.csv', help='input csv data')
    parser.add_argument('--output_data', type=str, default='data/CCLE_Mitelman_for_ML_numerical.csv',
                        help='output csv data with all columns converted into numerical columns')

    args = parser.parse_args()
    input_data = args.input_data
    output_data = args.output_data

    input_df = pd.read_csv(input_data)
    # drop the cell name column which does not contribute to training
    input_df.drop(columns=['dataset', 'Sample_ID'], inplace=True)
    print(input_df.shape)
    # there should be output of five columns: Y/N/P,ploidy_classification,DM amounts,HSR present,primary_or_metastasis
    print(input_df.select_dtypes(include=['object']))
    # Initialize the label encoder to convert categorical columns to numerical type
    label_encoder = LabelEncoder()
    # encode N->0, P->1, Y->2, Y/N/P->target
    input_df['target'] = label_encoder.fit_transform(input_df['ECDNA_classification'])
    input_df.drop(columns=['ECDNA_classification'], inplace=True)

    input_df['HSR_classification'] = label_encoder.fit_transform(input_df['HSR_classification'])

    # apply custom mapping to map ploidy_classification column to convert the column from strings to numbers, -1
    # means unknown
    input_df['ploidy_classification'] = map_feature(input_df['ploidy_classification'], ploidy_mapping)
    # convert XY_chromosomes column into numerical column
    input_df['XY_chromosomes'] = map_feature(input_df['XY_chromosomes'], chromosome_mapping)

    input_df.to_csv(output_data, index=False)
