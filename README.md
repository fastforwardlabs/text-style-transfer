# Text Style Transfer - Neutralizing Subjectivity Bias with Huggingface Transformers
This repo accompanies the FF24 research cycle focused on Text Style Transfer (TST). The contents of this repository support all exploratory and development work including:
- all data preprocessing steps and exploratory data analysis (EDA) on the Wiki Neutrality Corpus (WNC)
- all model training scripts and hyperparameter tuning experiments used while training both the BART seq2seq model for text style transfer, as well as the BERT classification model used for evaluating style transfer quality
- implementations of all custom evaluation metrics including Style Transfer Intensity (STI) & Content Preservation Score (CPS)
- evaluation processes for understanding performance of both models and custom metrics
- interactive notebooks that walk through the model/metric analysis, as well as demonstrations of using library functionality

## Repository Structure
```
.
├── data                                          # contains raw/post-processed data & eval metric artifacts
├── scripts                                       # contains all data prep, training, evaluation scripts
├── notebooks                                     # contains all notebooks that demonstrate basic usage
├── src                                           # main library of code supporting modeling and evalution
├── requirements.txt                              # python dependencies
├── LICENSE
└── README.md
```


### `src`

```
.
├── __init__.py                                   # 
├── data_utils.py                                 # helper functions used in data preprocessing scripts
├── inference.py                                  # implements SubjectivtyNeutralizer, StyleIntensityClassifier, ContentPreservationScorer
└── evaluation.py                                 # implements StyleTransferEvaluation, StyleClassifierEvaluation, ContentPreservationEvaluation
```
Please refer directly to the modules in this directory for detailed documentation on the purpose and functionality of each class/function.


### `scripts`

```
.
├── content_preservation_eval.py                  # experimental grid search to select CPS metric parameters
├── launch_tensorboard.py                         # for launching tensorboard as a CML application
├── prepare_data.py                               # builds seq2seq and classification datasets from WNC raw data
└── train
    ├── classifier              
    │   ├── hyperparameter_search                 # config + shell scripts for hyperparameter search experiments
    │   ├── train_classifier.py                   # custom train script for HuggingFace classifier fine-tuning
    │   └── train_classifier.sh                   # shell script for running train script with configs
    ├── seq2seq
    │   ├── train_seq2seq.py                      # custom/modified fine-tuning script for HuggingFace seq2seq
    │   └── train_seq2seq.sh                      # shell script for running train script with configs
    └── train_job.py                              # parent script used by CML Jobs to run either cls or seq2seq training
```


### `notebooks`

```
├── Covertype_EDA.ipynb                           # details data preprocessing and drift induction methods
├── Covtype_experiment_dev.ipynb                  # demonstration of how to interact with the test_harness library
└── archive                                       # directory full of working/development notebooks
```

Please refer to the individual notebooks for further info as they contain inline discussion, documentation, and analysis.

## Project Setup

This project was developed against Python 3.9.7.

Because the training scripts in this project (specifically the seq2seq training script) make use of a modified version of HuggingFace's `examples` fine-tuning script, we **must install the `transformers` library from source** with the same version of the repo as this project was initially developed on (4.18.0.dev0). For more info on installing HF from source, see [these docs](https://github.com/huggingface/transformers/tree/v4.18.0/examples).  After that, the other project requirements can be installed.

For ease of setup, an installation script has been provided that handles all of these details. Simply just run the `cdsw-build.sh` script from the root of the project. Additionally, you can then run `python3 prepare_data.py` to download the raw WNC files and create formatted HuggingFace datasets for classification and seq2seq modeling.