from typing import List

import torch
import numpy as np
import pandas as pd
from pyemd import emd
from transformers_interpret import SequenceClassificationExplainer
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)


class SubjectivityNeutralizer:
    """
    Seq2seq model wrapper used to generate neutral-toned text given a subjective text input.

    Attributes:
        model_identifier (str) - Path to the model that will be used by the pipeline to make predictions.
        max_gen_lenght (int) - upper limit on number of tokens the model can generate as output

    TO-DO:
        - Write tests to make sure each public method works...
        - Add option to show/return the diff between input/output

    """

    def __init__(self, model_identifier: str, max_gen_length: int = 200):
        self.model_identifier = model_identifier
        self.max_gen_length = max_gen_length
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        self._build_pipeline()

    def _build_pipeline(self):

        self.pipeline = pipeline(
            task="text2text-generation",
            model=self.model_identifier,
            device=self.device,
            max_length=self.max_gen_length,
        )

    def transfer(self, input_text: List[str]) -> List[str]:
        """
        Generate a neutral form of the provided text(s) while retaining
        the semantic meaning of input.

        Args:
            input_text (`str` or `List[str]`) - Input text for style transfer.

        Returns:
            generated_text - The generated text ouputs as a list of strings

        """
        return [item["generated_text"] for item in self.pipeline(input_text)]


class StyleIntensityClassifier:
    """
    Utility for calculating Style Transfer Intensity between
    two pieces of text (i.e. input and output of TST model)

    This custom evaluation metric aims to quantify the magnitude of transfered
    style between two texts. To accomplish this, we pass input and output texts
    through a trained style classifier to produce two distributions. We then
    utilize Earth Movers Distance (EMD) to calculate the minimum "cost"/"work"
    required to turn the input distribution into the output distribution. This
    metric allows us to capture a more nuanced, per-example measure of style
    transfer when compared to simply applying a classifier to determine the
    binary decision of achieved target style? (yes/no).

    Attributes:
        model_identifier (str)

    """

    def __init__(self, model_identifier: str):
        self.model_identifier = model_identifier
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        self._build_pipeline()

    def _build_pipeline(self):

        self.pipeline = pipeline(
            task="text-classification",
            model=self.model_identifier,
            device=self.device,
            return_all_scores=True,
        )

    def score(self, input_text: List[str]):
        """
        Classify a given input text as subjective or neutral using
        model initialized by the class.

        Args:
            input_text (`str` or `List[str]`) - Input text for classification

        Returns:
            classification (dict) - a dictionary containing the label, score, and
                distribution between classes

        """
        result = self.pipeline(input_text)
        distributions = np.array(
            [[label["score"] for label in item] for item in result]
        )
        return [
            {
                "label": self.pipeline.model.config.id2label[scores.argmax()],
                "score": scores.max(),
                "distribution": scores.tolist(),
            }
            for scores in distributions
        ]

    def calculate_transfer_intensity(
        self, input_text: List[str], output_text: List[str], target_class_idx: int = 1
    ) -> List[float]:
        """
        Calcualates the style transfer intensity (STI) between two pieces of text.

        Args:
            input_text (list) - list of input texts with indicies corresponding
                to counterpart in output_text
            ouptput_text (list) - list of output texts with indicies corresponding
                to counterpart in input_text
            target_class_idx (int) - index of the target style class used for directional
                score correction

        Returns:
            A list of floats with corresponding style transfer intensity scores.


        """

        if len(input_text) != len(output_text):
            raise ValueError(
                "input_text and output_text must be of same length with corresponding items"
            )

        input_dist = [item["distribution"] for item in self.score(input_text)]
        output_dist = [item["distribution"] for item in self.score(output_text)]

        return [
            self.calculate_emd(input_dist[i], output_dist[i], target_class_idx)
            for i in range(len(input_dist))
        ]

    @staticmethod
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


class ContentPreservationScorer:
    """
    Utility for calculating Content Preservation Score between
    two pieces of text (i.e. input and output of TST model)

    This custom evaluation metric aims to quantify content preservation by
    first modifying text to remove all style-related tokens leaving just
    content related tokens behind. Style tokens are determind on a
    sentence-by-sentence basis by extracting out salient token attributions
    from a trained Style Classifier (BERT) so contextual information is
    perserved in the attribution scores. Style tokens are then masked/removed
    from the text. We pass the style-less sentences through a pre-trained,
    but not fine-tuned SentenceBert model to compute sentence embeddings.
    Cosine similarity on the embeddings produces a score that should represent
    content preservation.

    PSUEDO-CODE: (higher score is better preservation)
    1. mask out style tokens for input and output text (1str)
    2. get SBERT embedddings for each (multi)
    3. calculate cosine similarity (multi pairs)

    Attributes:
        cls_model_identifier (str)
        sbert_model_identifier (str)

    """

    def __init__(self, cls_model_identifier: str, sbert_model_identifier: str):

        self.cls_model_identifier = cls_model_identifier
        self.sbert_model_identifier = sbert_model_identifier
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else -1

        self._initialize_hf_artifacts()

    def _initialize_hf_artifacts(self):
        """
        Initialize a HuggingFace artifacts (tokenizer and model) according
        to the provided identifiers for both SBert and the classification model.
        Then initialize the word attribution explainer with the HF model+tokenizer.

        """

        # sbert
        self.sbert_tokenizer = AutoTokenizer.from_pretrained(
            self.sbert_model_identifier
        )
        self.sbert_model = AutoModel.from_pretrained(self.sbert_model_identifier)

        # classifer
        self.cls_tokenizer = AutoTokenizer.from_pretrained(self.cls_model_identifier)
        self.cls_model = AutoModelForSequenceClassification.from_pretrained(
            self.cls_model_identifier
        )
        self.cls_model.to(self.device)
        self.explainer = SequenceClassificationExplainer(
            self.cls_model, self.cls_tokenizer
        )

    def compute_sentence_embeddings(self, input_text: List[str]) -> torch.Tensor:
        """
        Compute sentence embeddings for each sentence provided a list of text strings.

        Args:
            input_text (List[str]) - list of input sentences to encode

        Returns:
            sentence_embeddings (torch.Tensor)

        """
        # tokenize sentences
        encoded_input = self.sbert_tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        # to device
        self.sbert_model.eval()
        self.sbert_model.to(self.device)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # compute token embeddings
        with torch.no_grad():
            model_output = self.sbert_model(**encoded_input)

        return (
            self.mean_pooling(model_output, encoded_input["attention_mask"])
            .detach()
            .cpu()
        )

    def calculate_content_preservation_score(
        self,
        input_text: List[str],
        output_text: List[str],
        threshold: float = 0.3,
        mask_type: str = "pad",
        return_all: bool = False,
    ) -> List[float]:
        """
        Calcualates the content preservation score (CPS) between two pieces of text.

        Args:
            input_text (list) - list of input texts with indicies corresponding
                to counterpart in output_text
            ouptput_text (list) - list of output texts with indicies corresponding
                to counterpart in input_text
            return_all (bool) - If true, return dict containing intermediate
                text with style masking applied, along with scores
            mask_type (str) - "pad", "remove", or "none"

        Returns:
            A list of floats with corresponding content preservation scores.

        PSUEDO-CODE: (higher score is better preservation)
            1. mask out style tokens for input and output text (1str)
            2. get SBERT embedddings for each (multi)
            3. calculate cosine similarity (multi pairs)
        """
        if len(input_text) != len(output_text):
            raise ValueError(
                "input_text and output_text must be of same length with corresponding items"
            )

        if mask_type != "none":
            # Mask out style tokens
            masked_input_text = [
                self.mask_style_tokens(text, mask_type=mask_type, threshold=threshold)
                for text in input_text
            ]
            masked_output_text = [
                self.mask_style_tokens(text, mask_type=mask_type, threshold=threshold)
                for text in output_text
            ]

            # Compute SBert embeddings
            input_embeddings = self.compute_sentence_embeddings(masked_input_text)
            output_embeddings = self.compute_sentence_embeddings(masked_output_text)
        else:
            # Compute SBert embeddings on unmasked text
            input_embeddings = self.compute_sentence_embeddings(input_text)
            output_embeddings = self.compute_sentence_embeddings(output_text)

        # Calculate cosine similarity
        scores = self.cosine_similarity(input_embeddings, output_embeddings)

        if return_all:
            output = {
                "scores": scores,
                "masked_input_text": masked_input_text
                if mask_type != "none"
                else input_text,
                "masked_output_text": masked_output_text
                if mask_type != "none"
                else output_text,
            }
            return output
        else:
            return scores

    def calculate_feature_attribution_scores(
        self, text: str, class_index: int = 0, as_norm: bool = False
    ) -> List[tuple]:
        """
        Calcualte feature attributions using integrated gradients by passing
        a string of text as input.

        Args:
            text (str) - text to get attributions for
            class_index (int) - Optional output index to provide attributions for

        """
        attributions = self.explainer(text, index=class_index)

        if as_norm:
            return self.format_feature_attribution_scores(attributions)

        return attributions

    def mask_style_tokens(
        self,
        text: str,
        threshold: float = 0.3,
        mask_type: str = "pad",
        class_index: int = 0,
    ) -> str:
        """
        Utility function to mask out style tokens from a given string of text.

        Style tokens are determined by first calculating feature importances (via
        word attributions from trained StyleClassifer) for each token in the input sentence.
        We then normalize the absolute values of attributions scores to see how much each token
        contributes as a percentage overall style classification and rank those in descending order.

        We then select the top N tokens that account for the cumulative _threshold_ amount (%) of
        total styleattribution. By using cumulative percentages, N is not a fixed number and we
        ultimately take however many tokens are needed to account for _threshold_ % of the overall
        style.

        We can optionally return a string with these style tokens padded out or completely removed
        by toggling _mask_type_ between "pad" and "remove".

        Args:
            text (str)
            threshold (float) - percentage of style attribution as cutoff for masking selection.
            mask_type (str) - "pad" or "remove", indicates how to handle style tokens
            class_index (str)

        Returns:
            text (str)

        """

        # get attributions and format as sorted dataframe
        attributions = self.calculate_feature_attribution_scores(
            text, class_index=class_index, as_norm=False
        )
        attributions_df = self.format_feature_attribution_scores(attributions)

        # select tokens to mask
        token_idxs_to_mask = []

        # If the first token accounts for more than the set
        # threshold, take just that token to mask. Otherwise,
        # take all tokens up to the threshold
        if attributions_df.iloc[0]["cumulative"] > threshold:
            token_idxs_to_mask.append(attributions_df.index[0])
        else:
            token_idxs_to_mask.extend(
                attributions_df[
                    attributions_df["cumulative"] <= threshold
                ].index.to_list()
            )

        # Build text sequence with tokens masked out
        mask_map = {"pad": "[PAD]", "remove": ""}
        toks = [token for token, score in attributions]
        for idx in token_idxs_to_mask:
            toks[idx] = mask_map[mask_type]

        if mask_type == "remove":
            toks = [token for token in toks if token != ""]

        # Decode that sequence
        masked_text = self.explainer.tokenizer.decode(
            self.explainer.tokenizer.convert_tokens_to_ids(toks),
            skip_special_tokens=False,
        )

        # Remove special characters other than [PAD]
        for special_token in self.explainer.tokenizer.all_special_tokens:
            if special_token != "[PAD]":
                masked_text = masked_text.replace(special_token, "")

        return masked_text.strip()

    def visualize_feature_attribution_scores(self, text: str, class_index: int = 0):
        """
        Calculates and visualizes feature attributions using integrated gradients.

        Args:
            text (str) - text to get attributions for
            class_index (int) - Optional output index to provide attributions for

        """
        self.explainer(text, index=class_index)
        self.explainer.visualize()

    @staticmethod
    def format_feature_attribution_scores(attributions: List[tuple]) -> pd.DataFrame:
        """
        Utility for formatting attribution scores for style token mask selection

        Sorts a given List[tuple] where tuples represent (token, score) by the
        normalized absolute value of each token score.

        """

        df = pd.DataFrame(attributions, columns=["token", "score"])
        df["abs_norm"] = df["score"].abs() / df["score"].abs().sum()
        df = df.sort_values(by="abs_norm", ascending=False)
        df["cumulative"] = df["abs_norm"].cumsum()
        return df

    @staticmethod
    def cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> List[float]:
        """
        Calculate cosine similarity on pairs of embedddings.

        Can handle 1D Tensor for single pair or 2D Tensors with corresponding indicies
        for matrix operation on multiple pairs.

        """

        assert tensor1.shape == tensor2.shape

        # ensure 2D tensor
        if tensor1.ndim == 1:
            tensor1 = tensor1.unsqueeze(0)
            tensor2 = tensor2.unsqueeze(0)

        cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos_sim(tensor1, tensor2).tolist()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Peform mean pooling over token embeddings to create sentence embedding. Here we take
        the attention mask into account for correct averaging on active token positions.

        CODE BORROWED FROM:
            https://www.sbert.net/examples/applications/computing-embeddings/README.html#sentence-embeddings-with-transformers

        """

        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask
