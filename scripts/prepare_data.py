import os

from src.data_utils import build_hf_dataset, remove_duplicate_by_revid, remove_outliers

RAW_ROOT_PATH = 'data/raw'
RAW_DATA_PATH = os.path.join(RAW_ROOT_PATH, "bias_data/WNC")
TARGET_PATH = os.path.join(os.path.dirname(RAW_ROOT_PATH), "processed")


if __name__ == "__main__":
    
    os.makedirs(RAW_ROOT_PATH, exist_ok=True)
    os.makedirs(TARGET_PATH, exist_ok=True)
    
    # download data if needed
    if len(os.listdir(RAW_ROOT_PATH)) == 0:
        os.system(f"!curl -L http://nlp.stanford.edu/projects/bias/bias_data.zip -o {RAW_ROOT_PATH}/bias_data.zip && !unzip {RAW_ROOT_PATH}/bias_data.zip -d {RAW_ROOT_PATH}")
        
    # preprocess data
    wnc_datasets = build_hf_dataset(RAW_DATA_PATH)
    wnc_datasets = remove_duplicate_by_revid(wnc_datasets)
    wnc_datasets = remove_outliers(wnc_datasets)
    wnc_datasets = wnc_datasets.remove_columns(["length_pre", "length_post", "length_delta"])
    
    # save one word version DatasetDict as apache arrow table
    WORD_VERSION_PATH = os.path.join(TARGET_PATH, 'one-word-version')
    os.makedirs(WORD_VERSION_PATH)
    wnc_datasets.save_to_disk(WORD_VERSION_PATH)