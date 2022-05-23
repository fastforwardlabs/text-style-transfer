from typing import List

import torch
import numpy as np
from transformers import pipeline

from src.evaluation import calculate_emd


class SubjectivityNeutralizer:
    """
    Seq2seq model wrapper used to generate neutral-toned text given a subjective text input.

    Attributes:
        model_identifier (str) - Path to the model that will be used by the pipeline to make predictions.


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

    def transfer(self, input_text):
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

    TO-DO:
        - Add option classify text as neutral/subjective
        - Add ability to calculate STI
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

    def score(self, input_text):
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
            calculate_emd(input_dist[i], output_dist[i], target_class_idx)
            for i in range(len(input_dist))
        ]
