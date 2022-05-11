import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


from collections import defaultdict
from datasets import (
    load_dataset,
    load_from_disk,
    load_metric,
    Dataset,
    Features,
    Value,
    ClassLabel,
    DatasetDict,
)
from transformers.integrations import MLflowCallback
from transformers.trainer_utils import IntervalStrategy
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed, 
    HfArgumentParser
)



@dataclass
class StiTrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop itself**. Here we are selecting only those relevant to our Style Transfer Intensity classification problem.
    
    Using [`HfArgumentParser`] we can turn this class into [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the command line.
    """
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    logging_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    evaluation_strategy: IntervalStrategy = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )

@dataclass
class MiscArguments:
    """
    Additional modeling arguments that are not a direct part of `transformers.TrainingArguments`.
    
    """
    
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


def main():
    
    set_seed(42)
    
    # establish training arguments 
    parser = HfArgumentParser((StiTrainingArguments, MiscArguments))
    sti_args, misc_args = parser.parse_args_into_dataclasses()
    training_args = TrainingArguments(**vars(sti_args))
    
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
        
    print("TEST!!!!", misc_args.model_name_or_path)
    
    # load WNC classification dataset
    CLS_DATASET_PATH = "/home/cdsw/data/processed/WNC_full_cls"
    wnc_full_cls = load_from_disk(CLS_DATASET_PATH)
    
    # load base-model and tokenizer
    checkpoint = misc_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    tokenized_datasets = wnc_full_cls.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    def compute_metrics(eval_preds):

        accuracy_metric = load_metric("accuracy")
        f1_metric = load_metric("f1")

        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_metric.compute(predictions=predictions, references=labels),
            "f1": f1_metric.compute(predictions=predictions, references=labels),
        }

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.remove_callback(MLflowCallback)

    # trainer.train()
    
if __name__ == "__main__":
    main()