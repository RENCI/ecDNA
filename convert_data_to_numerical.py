import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_data', type=str, default='data/ec_master.csv', help='input csv data')
    parser.add_argument('--output_data', type=str, default='data/ec_master_numerical.csv', help='input csv data')

    args = parser.parse_args()
    input_data = args.input_data
    output_data = args.output_data

    input_df = pd.read_csv(input_data)
    # drop the cell name column which does not contribute to training
    input_df.drop(columns=['CCLE_Name', 'DM amounts'], inplace=True)
    print(input_df.shape)
    # there should be output of five columns: Y/N/P,ploidy_classification,DM amounts,HSR present,primary_or_metastasis
    print(input_df.select_dtypes(include=['object']))
    # Initialize the label encoder to convert categorical columns to numerical type
    label_encoder = LabelEncoder()
    # encode N->0, P->1, Y->2, Y/N/P->target
    input_df['target'] = label_encoder.fit_transform(input_df['Y/N/P'])
    input_df.drop(columns=['Y/N/P'], inplace=True)

    # apply custom mapping to map ploidy_classification column to convert the column from strings to numbers, -1
    # means unknown
    mapping = {
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
    input_df['ploidy_classification'] = input_df['ploidy_classification'].map(mapping)
    input_df['ploidy_classification'] = input_df['ploidy_classification'].fillna(-1)
    input_df['ploidy_classification'] = input_df['ploidy_classification'].astype('Int8')

    # about 94% of HSR present column has NaN/unknown, so cannot impute this column. Encode them as N->0, Y->1, NaN->-1
    mapping = {
        'Y': 1,
        'N': 0
    }
    input_df['HSR present'] = input_df['HSR present'].map(mapping)
    input_df['HSR present'] = input_df['HSR present'].fillna(-1)
    input_df['HSR present'] = input_df['HSR present'].astype('Int8')

    # for the primary_or_metastasis column, there is 293 records in Primary category, 231 in Metastasis category,
    # and 192 unknowns
    mapping = {
        'Primary': 1,
        'Metastasis': 2
    }
    input_df['primary_or_metastasis'] = input_df['primary_or_metastasis'].map(mapping)
    input_df['primary_or_metastasis'].fillna(-1)
    input_df['primary_or_metastasis'] = input_df['primary_or_metastasis'].astype('Int8')

    input_df.to_csv(output_data, index=False)
