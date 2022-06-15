import os
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src.inference import (
    SubjectivityNeutralizer,
    StyleIntensityClassifier,
    ContentPreservationScorer,
)


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

            if kind == "fp":
                print(
                    f"Record #{row.name} classified as {pred_label}, when actual label is {actual_label}: \n\t",
                    self.metric_df.loc[subj_idx].text,
                )
                print()
                print(
                    f"Here's its {pred_label} labeled counterpart #{neut_idx}: \n\t",
                    self.metric_df.loc[neut_idx].text,
                )
                print(
                    "---------------------------------------------------------------------"
                )
                print()
            else:
                print(
                    f"Record #{row.name} classified as {pred_label}, when actual label is {actual_label}: \n\t",
                    self.metric_df.loc[neut_idx].text,
                )
                print()
                print(
                    f"Here's its {pred_label} labeled counterpart #{subj_idx}: \n\t",
                    self.metric_df.loc[subj_idx].text,
                )
                print(
                    "---------------------------------------------------------------------"
                )
                print()


class ContentPreservationEvaluation:
    """
    A utility class for evaluating our custom Contentent Preservation metric.

    After initializing the class with a style classification model path, SentenceBERT model
    path, and dataset identifier, this class will perform evalution on the validation split
    and save out metrics that allow for detailed error analysis.

    Attributes:
        cls_model_identifier (str)
        sbert_model_identifier (str)
        dataset_identifier (str)
        threshold (float)
        mask_type (str)

    """

    def __init__(
        self,
        cls_model_identifier: str,
        sbert_model_identifier: str,
        dataset_identifier: str,
        threshold: float,
        mask_type: str,
    ):

        self.cls_model_identifier = cls_model_identifier
        self.sbert_model_identifier = sbert_model_identifier
        self.dataset_identifier = dataset_identifier
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.threshold = threshold
        self.mask_type = mask_type

        self._initialize_hf_artifacts()
        self._construct_dataloader()

    def _initialize_hf_artifacts(self):
        """
        Initialize a HuggingFace dataset and ContentPreservationScorer
        according for inference using the provided identifiers.

        """
        self.dataset = load_from_disk(self.dataset_identifier)
        self.cps = ContentPreservationScorer(
            sbert_model_identifier=self.sbert_model_identifier,
            cls_model_identifier=self.cls_model_identifier,
        )

    def _construct_dataloader(self):
        """
        Initialize the evaluation dataloader.

        Note: here we are batching untokenized sentences since the downstream evaluation
        pipeline metrics will operate on raw text as input.

        """
        self.eval_dataloader = torch.utils.data.DataLoader(
            self.dataset["validation"],
            batch_size=32,
            drop_last=False,
            pin_memory=True,
            shuffle=False,
        )

    def evaluate(self, save_name="metric_df"):
        """
        Content Preservation Evaluation.

        """

        metric_collection = defaultdict(list)

        for batch in tqdm(self.eval_dataloader):

            cps_output = self.cps.calculate_content_preservation_score(
                input_text=batch["source_text"],
                output_text=batch["target_text"],
                threshold=self.threshold,
                mask_type=self.mask_type,
                return_all=True,
            )

            batch["masked_source_text"] = cps_output["masked_input_text"]
            batch["masked_target_text"] = cps_output["masked_output_text"]
            batch["cps_score"] = cps_output["scores"]

            # collect data
            for k, v in batch.items():
                metric_collection[k].extend(v)

        self.metric_df = pd.DataFrame(metric_collection)
        self._save_metric_df(f"{self.mask_type}-{self.threshold}-{save_name}")

    def _save_metric_df(self, name):

        FILE_PATH = os.path.join(
            os.path.expanduser("~"), "data/output/cpe_metrics/", f"{name}.pkl"
        )
        os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
        self.metric_df.to_pickle(FILE_PATH)

        print(f"Saved `self.metric_df` to {FILE_PATH}")

    def load_metric_df(self, name="metric_df"):

        FILE_PATH = os.path.join(
            os.path.expanduser("~"), "data/output/cpe_metrics/", f"{name}.pkl"
        )
        self.metric_df = pd.read_pickle(FILE_PATH)

        print(f"Loaded `self.metric_df` from {FILE_PATH}")


class StyleTransferEvaluation:
    """

    TO-DO:
        - Build caching system for pred_text and metric_df bbased on model/dataset identifiers
    """

    def __init__(
        self,
        seq2seq_model_identifier: str,
        cls_model_identifier: str,
        dataset_identifier: str,
    ):
        self.seq2seq_model_identifier = seq2seq_model_identifier
        self.cls_model_identifier = cls_model_identifier
        self.dataset_identifier = dataset_identifier
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self._initialize_hf_artifacts()
        self._construct_dataloader()

    def _initialize_hf_artifacts(self):
        """
        Initialize a HuggingFace dataset, SubjectivityNeutralizer, StyleIntensityClassifier,
        and ContentPreservationScorer inference using the provided identifiers.

        """
        self.dataset = load_from_disk(self.dataset_identifier)
        self.sn = SubjectivityNeutralizer(self.seq2seq_model_identifier)
        self.sc = StyleIntensityClassifier(self.cls_model_identifier)

    def _construct_dataloader(self):
        """
        Initialize the evaluation dataloader.

        Note: here we are batching untokenized sentences since the downstream evaluation
        pipeline metrics will operate on raw text as input.

        """
        self.eval_dataloader = torch.utils.data.DataLoader(
            self.dataset["validation"],
            batch_size=32,
            drop_last=False,
            pin_memory=True,
            shuffle=False,
        )

    def evaluate(self, save_name="metric_df"):
        """
        Peformamce evaluation.

        """

        metric_collection = defaultdict(list)

        for batch in tqdm(self.eval_dataloader):

            batch["pred_text"] = self.sn.transfer(batch["source_text"])
            batch["pred_sti"] = self.sc.calculate_transfer_intensity(
                input_text=batch["source_text"], output_text=batch["pred_text"]
            )
            batch["target_sti"] = self.sc.calculate_transfer_intensity(
                input_text=batch["source_text"], output_text=batch["target_text"]
            )

            # collect data
            for k, v in batch.items():
                metric_collection[k].extend(v)

        self.metric_df = pd.DataFrame(metric_collection)
        self._save_metric_df(save_name)

    def _save_metric_df(self, name):

        FILE_PATH = os.path.join(
            os.path.expanduser("~"), "data/output/ste_metrics/", f"{name}.pkl"
        )
        os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
        self.metric_df.to_pickle(FILE_PATH)

        print(f"Saved `self.metric_df` to {FILE_PATH}")

    def load_metric_df(self, name="metric_df"):

        FILE_PATH = os.path.join(
            os.path.expanduser("~"), "data/output/ste_metrics/", f"{name}.pkl"
        )
        self.metric_df = pd.read_pickle(FILE_PATH)

        print(f"Loaded `self.metric_df` from {FILE_PATH}")
