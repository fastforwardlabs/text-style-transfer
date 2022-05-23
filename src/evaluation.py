from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from pyemd import emd
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def calculate_emd(input_dist, output_dist, target_class_idx):
    """
    Calculate the direction-corrected Earth Mover's Distance (aka Wasserstein distance)
    between two distributions of equal length. Here we penalize the EMD score if
    the output text style moved further away from the target style.

    Reference: https://github.com/passeul/style-transfer-model-evaluation/blob/master/code/style_transfer_intensity.py

    Args:
        input_dist (list) - probabilities assigned to the style classes
            from the input text to style transfer model
        output_dist (list) - probabilities assigned to the style classes
            from the outut text of the style transfer model

    Returns:
        emd (float) - Earth Movers Distance between the two distributions

    """

    N = len(input_dist)
    distance_matrix = np.ones((N, N))
    dist = emd(np.array(input_dist), np.array(output_dist), distance_matrix)

    transfer_direction_correction = (
        1 if output_dist[target_class_idx] >= input_dist[target_class_idx] else -1
    )

    return round(dist * transfer_direction_correction, 4)


class StyleClassifierEvaluation:
    """
    A utility class for evaluating Style Classification models (binary classification).

    After initializing the class with a style classification model path and dataset identifier,
    this class will perform evalution on the validation split and save out metrics that allow
    for detailed error analysis inluding: confusion matrix, classification report, and inspection
    of severe false positive and false negative predictions.


    TO-DO:
        - generalize this to specificy or loop over both `validation` and `test` datasets
        - comparative metrics between the two datasets

    """

    def __init__(
        self, model_identifier: str, dataset_identifier: str, num_labels: int = 2
    ):
        self.model_identifier = model_identifier
        self.dataset_identifier = dataset_identifier
        self.num_labels = num_labels
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self._initialize_hf_artifacts()
        self._prepare_data()
        self._construct_dataloader()

    def _initialize_hf_artifacts(self):
        """
        Initialize a HuggingFace dataset, tokenizer, model, and data_collator according
        to the provided identifiers.

        """
        self.dataset = load_from_disk(self.dataset_identifier)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_identifier)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_identifier, num_labels=self.num_labels
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _prepare_data(self):
        """
        Tokenize the instantiated dataset then cleanup columns to include only
        required fields with proper names.

        """

        self.tokenized_dataset = self.dataset.map(self._tokenize_function, batched=True)

        self.tokenized_dataset = self.tokenized_dataset.remove_columns(
            ["text"]
        ).rename_column("label", "labels")
        self.tokenized_dataset.set_format("torch")

    def _construct_dataloader(self):
        """
        Initialize the evaluation dataloader.

        """
        self.eval_dataloader = torch.utils.data.DataLoader(
            self.tokenized_dataset["validation"],
            batch_size=8,
            collate_fn=self.data_collator,
            drop_last=False,
            pin_memory=True,
            shuffle=False,
        )

    def _tokenize_function(self, example):
        return self.tokenizer(example["text"], truncation=True)

    def evaluate(self):
        """
        Peformamce evaluation

        Using the specified model and dataset, this method gathers predictions
        and ground truth values, then saves out all results and corresponding `text`
        to the `metric_df` class attribute.

        """

        metric_collection = defaultdict(list)

        self.model.eval()
        self.model.to(self.device)
        for batch in tqdm(self.eval_dataloader):

            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)

            # post-process predictions
            logits = outputs.logits
            preds = torch.nn.functional.softmax(logits, dim=-1)
            pred_labels = torch.argmax(preds, dim=-1)
            pred_scores = preds[range(preds.shape[0]), pred_labels]

            # collect as metrics
            metric_collection["true_label"].extend(batch["labels"].tolist())
            metric_collection["pred_label"].extend(pred_labels.tolist())
            metric_collection["pred_score"].extend(pred_scores.tolist())

        self.metric_df = pd.DataFrame(metric_collection)
        self.metric_df.insert(
            loc=0, column="text", value=self.dataset["validation"]["text"]
        )

    def plot_confusion_matrix(self):
        """
        Generate and return a ConfustionMatrixDisplay for
        the saved `metrics_df` attribute.

        """
        return ConfusionMatrixDisplay.from_predictions(
            y_true=self.metric_df.true_label,
            y_pred=self.metric_df.pred_label,
            cmap=plt.cm.Blues,
        )

    def get_classification_report(self):
        """
        Generate and return a classification report for
        the saved `metrics_df` attribute.

        """
        return pd.DataFrame(
            classification_report(
                y_true=self.metric_df.true_label,
                y_pred=self.metric_df.pred_label,
                output_dict=True,
            )
        ).T.round(3)

    def highlight_classification_errors(self, kind: str = "fp", n: int = 10):
        """
        Investigate examples where classifier was strongly confident and wrong.

        This function queries the `metric_df` attribute for false positives or false
        negatives sorted by strongest classification score. It then prints out the
        top `n` examples in either case. Use this function to perform detailed error
        analysis and look for patterns.

        Args:
            kind (str) - `fn` (false negative) or `fp` (false positive) to inspect
            n (int) - number of instances to look at

        """

        if kind not in ["fp", "fn"]:
            raise ValueError(
                "Must specify value of `fn` (false negative) or `fp` (false positive)"
            )

        kind_map = {"fp": {"truth": 0, "pred": 1}, "fn": {"truth": 1, "pred": 0}}

        errors = self.metric_df[
            (self.metric_df.true_label == kind_map[kind]["truth"])
            & (self.metric_df.pred_label == kind_map[kind]["pred"])
        ].sort_values(by="pred_score", ascending=False)

        pred_label = "NEUTRAL" if kind == "fp" else "SUBJECTIVE"
        actual_label = "SUBJECTIVE" if kind == "fp" else "NEUTRAL"

        for _, row in errors[:n].iterrows():

            if row.name % 2 == 0:
                subj_idx = row.name
                neut_idx = row.name + 1
            else:
                subj_idx = row.name - 1
                neut_idx = row.name

            print(
                f"Record #{row.name} classified as {pred_label}, when actually {actual_label}: \n\t",
                self.metric_df.loc[subj_idx].text,
            )
            print()
            print(
                f"Here's its {pred_label} counterpart #{neut_idx}: \n\t",
                self.metric_df.loc[neut_idx].text,
            )
            print(
                "---------------------------------------------------------------------"
            )
            print()
