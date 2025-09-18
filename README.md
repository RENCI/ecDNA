# Development of machine learning models for classification and analysis of Extrachromosomal DNA (ecDNA) in cancer

This repository includes source code and initial analysis results for applying machine learning approaches 
to classify and analyse ecDNA in cancer using a gold standard database CytoCellDB. 
It represents our initial effort to assess the application of CytoCellDB data in the 
classification and analysis of ecDNA using machine learning approaches. Our initial findings were 
published in [NAR Cancer](https://academic.oup.com/narcancer). For full details, please refer to our paper 
[CytoCellDB: a comprehensive resource for exploring extrachromosomal DNA in cancer cell lines](https://pmc.ncbi.nlm.nih.gov/articles/PMC11292414/). 
We continue to enhance CytoCellDB by expanding the data and refining our machine learning models, 
and this work is ongoing.

## Installation and Environment Setup for running model training, prediction, and analysis code

The code for model training, prediction, and analysis in this repo is written in 
[python](https://www.python.org/), so you will need to have python version 3.xx installed. We suggest 
to create a [conda](https://docs.conda.io/en/latest/) environment or a 
[python virtual environment](https://docs.python.org/3/library/venv.html) to better manage module version 
dependencies. To create a conda environment, follow the instructions 
[here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment). 
To create a python virtual environment, follow the instructions [here](https://docs.python.org/3/library/venv.html).

After you set up a conda environment or a python virtual environment, you need to activate the environment, then 
follow the steps below to install dependencies and run code for model training, prediction, and analysis.

- Clone the repo by running `git clone https://github.com/RENCI/ecDNA.git`.
- Make sure you activate your conda or virtual environment and then run the command below to install dependencies.
```
cd ecDNA
pip install --no-cache-dir -r requirements.txt
```
- Change the directory to ecDNA/src source code directory: `cd src`
- Run the scripts in the source code directory as needed. Make sure you have input data ready to be processed 
by the scripts. For example, you can create a `data` directory under `ecDNA` directory and put the input csv file 
`df_CCLE_Mitelman_for_ML.csv` in `ecDNA/data` directory, which is the default input data set up in the scripts, 
but you can put input data anywhere and pass the data with path as an input argument to the scripts. The comments 
on the top of each script in this source code directory details the functionality of each script along with 
which step in the data processing workflow this scipt should be run. In general, the scripts should be run 
in the order of data processing workflow sequence as shown below:
  - Process the initial input data to remove all-null columns and all rows with null in target column. Change the input 
  parameters to whatever you want that fit how you want your data to be organized.
  ```
    python src/process_data.py --input_data data/df_CCLE_Mitelman_for_ML.csv --output_data  data/CCLE_Mitelman_for_ML.csv --classification_column ECDNA_classification
  ```
  - Convert all non-numerical columns in the processed data from the previous step to numerical types.
  ```
    python src/convert_data_to_numerical.py --input_data data/CCLE_Mitelman_for_ML.csv --output_data data/CCLE_Mitelman_for_ML_numerical.csv
  ```
  - Impute all columns with missing data in the numerical data obtained from the previous step.
  ```
    python src/impute_features.py --input_data data/CCLE_Mitelman_for_ML_numerical.csv --output_data data/CCLE_Mitelman_for_ML_imputed.csv
  ```
  - Train a classifier model using the imputed data from the previous step. It does feature scaling first 
before training to make sure all features have the same scale. The supported model_type parameters 
include `Gradient Boosting`, `Random Forest`, `decision_tree`, and `svm`. You can also pass in 
`--hsr_target` to the command below to add hsr_classification into target as well to train 
a model for two label classification to classify both ecDNA and HSR to differentiate them.
  ```
    python src/train.py --input_data data/CCLE_Mitelman_for_ML_imputed.csv --model_type 'Gradient Boosting' --output_model 'model_data/gradient_boosting_2_labels_model.joblib' --output_analysis_data analysis_data/gradient_boosting_2_labels_features.csv 
  ```
  - Train a bagging and stacking ensemble classifier model with GradientBoosting and RandomForest classifiers 
  as base and stacking estimators. 
  ```
    python src/ensemble_train.py --input_data data/CCLE_Mitelman_for_ML_imputed.csv --output_model model_data/ensemble_model.joblib
  ```
  - Evaluate a trained classifier model using the data with P or probable target which were 
not used for training and using the whole training data as well.
  ```
    python src/evaluate.py --ori_input_data data/CCLE_Mitelman_for_ML.csv --input_data data/CCLE_Mitelman_for_ML_imputed.csv --input_model model_data/gradient_boosting_2_labels_model.joblib --hsr_target --output_data data/CCLE_Mitelman_gradient_boosting_2_labels_p_predicted.csv 
  ```
  - Use a trained classifier model to make inference on new data beyond training and validation data.
  ```
    python src/infer.py --input_data data/Mouse_karyotypes_ECDNA.csv --train_data data/CCLE_Mitelman_for_ML_imputed.csv --input_model model_data/gradient_boosting_model.joblib --output_data data/Mouse_karyotypes_gradient_boosting_predicted.csv 
  ```
Note that trained Gradient Boosting and Random Forest classifier models have the best performance for our ecDNA and HSR classification.
