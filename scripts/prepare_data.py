import os

from src.data_utils import construct_seq2seq_dataset, construct_classification_dataset, remove_duplicate_by_revid, remove_outliers

RAW_ROOT_PATH = "data/raw"
RAW_DATA_PATH = os.path.join(RAW_ROOT_PATH, "bias_data/WNC")
TARGET_PATH = os.path.join(os.path.dirname(RAW_ROOT_PATH), "processed")

def build_seq2seq_data():
    """
    Format, preprocess, and save the WNC corpus as HuggingFace datasets to use for
    training a seq2seq text style transfer model.
    
    """
    
    versions = [
        {"name": "one_word", "bool_arg": True},
        {"name": "full", "bool_arg": False},
    ]

    for version in versions:

        print(f'Preparing {version["name"]} version of WNC dataset for seq2seq.')

        # preprocess data
        wnc_datasets = construct_seq2seq_dataset(RAW_DATA_PATH, one_word=version["bool_arg"])
        wnc_datasets = remove_duplicate_by_revid(wnc_datasets)
        wnc_datasets = remove_outliers(wnc_datasets, one_word=version["bool_arg"])
        wnc_datasets = wnc_datasets.map(
            lambda example: {
                "source_text": example["translation"]["pre"],
                "target_text": example["translation"]["post"],
            }
        )
        wnc_datasets = wnc_datasets.remove_columns(
            ["length_pre", "length_post", "length_delta", "translation"]
        )
        wnc_datasets["validation"] = wnc_datasets.pop("dev")

        # save version of DatasetDict as apache arrow table
        VERSION_PATH = os.path.join(TARGET_PATH, f"WNC_seq2seq_{version['name']}")
        os.makedirs(VERSION_PATH, exist_ok=True)
        wnc_datasets.save_to_disk(VERSION_PATH)

        print(f'Finished preparing {version["name"]} version of WNC dataset for seq2seq.')
    

def build_cls_data():
    """
    Format and save the WNC corpus as HuggingFace datasets to use for
    training a style classification model.
    
    """
    
    print(f'Preparing WNC dataset for classification.')
    
    construct_classification_dataset(path=os.path.join(TARGET_PATH, "WNC_seq2seq_full"))
    
    print(f'Finished preparing WNC dataset for classification.')



if __name__ == "__main__":

    os.makedirs(RAW_ROOT_PATH, exist_ok=True)
    os.makedirs(TARGET_PATH, exist_ok=True)

    # download data if needed
    if len(os.listdir(RAW_ROOT_PATH)) == 0:
        os.system(
            f"curl -L http://nlp.stanford.edu/projects/bias/bias_data.zip -o {RAW_ROOT_PATH}/bias_data.zip && unzip {RAW_ROOT_PATH}/bias_data.zip -d {RAW_ROOT_PATH}"
        )
        
    # build translation/seq2seq formatted datasets
    build_seq2seq_data()
    
    # build style classification dataset
    build_cls_data()