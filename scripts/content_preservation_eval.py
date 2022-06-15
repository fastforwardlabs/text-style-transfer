import os
from src.evaluation import ContentPreservationEvaluation

SBERT_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
CLS_MODEL_PATH = (
    "/home/cdsw/models/TRIAL-J-shuffle-lr_3en06-epoch_15-wd_.1-bs_32/checkpoint-67466"
)
DATASETS_PATH = "/home/cdsw/data/processed/WNC_seq2seq_full"

print(os.getcwd())

# threshold and mask_type experiments
for threshold in [t / 10 for t in range(1, 6, 1)]:
    for mask_type in ["pad", "remove"]:
        cpe = ContentPreservationEvaluation(
            sbert_model_identifier=SBERT_MODEL_PATH,
            cls_model_identifier=CLS_MODEL_PATH,
            dataset_identifier=DATASETS_PATH,
            threshold=threshold,
            mask_type=mask_type,
        )
        cpe.evaluate()
        
        print(f"Finished experiment: {mask_type}, {threshold}")
        

# baseline experiment
cpe = ContentPreservationEvaluation(
            sbert_model_identifier=SBERT_MODEL_PATH,
            cls_model_identifier=CLS_MODEL_PATH,
            dataset_identifier=DATASETS_PATH,
            threshold=0,
            mask_type="none"
        )
cpe.evaluate()