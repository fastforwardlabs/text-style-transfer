import os
from functools import reduce
from operator import concat
from collections import defaultdict

from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict, Value, Translation, Features


def build_hf_dataset(path: str) -> DatasetDict:
    """
    Formats the raw biased-word data (train/dev/test) into a HuggingFace DatasetDict object.

    Provided a path to the raw, one-word Wiki Neutrality Corpus (WNC) data files, this function parses
    each of the dev, test, and train sets and formats them as a HuggingFace DatasetDict object
    in the seq2seq style (i.e. ready for translation tasks).

    For data, see -> https://arxiv.org/pdf/1911.09709.pdf

    Args:
        path (str): path to directory containing raw WNC data files

    Returns:
        DatasetDict

    """

    splits = ["dev", "test", "train"]
    dataset_dict = defaultdict(dict)

    FEATURES = Features(
        {
            "rev_id": Value("string"),
            "translation": Translation(languages=["pre", "post"]),
        }
    )

    for split in splits:

        PATH = os.path.join(path, f"biased.word.{split}")

        rev_ids = []
        translation_pairs = []
            
        with open(PATH) as f:
            for i, line in enumerate(tqdm(f)):
                parts = line.strip().split("\t")

                # note some entries contain the POS and REL fields, others dont
                if len(parts) == 7:
                    rev_id, pre_tok, post_tok, pre_raw, post_raw, pos, rels = parts

                elif len(parts) == 5:
                    rev_id, pre_tok, post_tok, pre_raw, post_raw = parts

                else:
                    print(f"Skipped entry: {i}")

                rev_ids.append(rev_id)
                translation_pairs.append({"pre": pre_raw, "post": post_raw})

        split_dict = {
            "rev_id": rev_ids,
            "translation": translation_pairs,
        }

        dataset_dict[split] = Dataset.from_dict(split_dict, features=FEATURES)

    return DatasetDict(dataset_dict)


def remove_duplicate_by_revid(datasets: DatasetDict) -> DatasetDict:
    """
    Remove duplicate records from datasets.

    Provided a DatasetDict of the WNC, this function filters out all duplicate
    records as defined by their rev_id.

    Args:
        datasets (DatasetDict): a WNC datasets object returned by `build_hf_dataset()`

    Returns:
        DatasetDict

    """

    rev_ids = reduce(concat, [datasets[split]["rev_id"] for split in datasets.keys()])

    duplicate_revids = set([rev_id for rev_id in rev_ids if rev_ids.count(rev_id) > 1])

    print(f"{len(duplicate_revids)*2} duplicate records have been removed.")

    return datasets.filter(lambda x: x["rev_id"] not in duplicate_revids)


def remove_outliers(datasets: DatasetDict) -> DatasetDict:
    """
    Remove outlier observations from the datasets.

    Provided a DatasetDict of the WNC, this function filters out all outlier
    records as defined by:

        1. Records with pre-edit sentence length > 99th percentile.
        2. Records with pre-edit sentence length < 1st percentile.
        3. Records with net subtraction of more than 1 word. This is a "one word edit"
            version of the dataset so these are here in error. Manual inspections show
            these are in here due to grammer edits that occured along with NPOV edit.
        4. Records with net addition of more than 4 terms. This is based on manual
            inspection of the one-word version where additive edits up to net of 4 tokens
            seem to be mostly legitimate NPOV replacements, while those above 4 are rare
            and typically include a grammatical edit along with the NPOV edit (i.e. commas
            with extra spaces).

    Args:
        datasets (DatasetDict): a WNC datasets object returned by `build_hf_dataset()`

    Returns:
        DatasetDict

    """

    # first create new columns with calculated sentence lengths and the difference
    datasets = datasets.map(
        lambda example: {
            "length_pre": len(example["translation"]["pre"].split()),
            "length_post": len(example["translation"]["post"].split()),
        }
    ).map(
        lambda example: {"length_delta": example["length_post"] - example["length_pre"]}
    )

    # set length difference bounds determine sentence length cutoffs
    subtractive_bound = -1
    additive_bound = 4
    lower_length, upper_length = (
        datasets["train"].to_pandas()["length_pre"].quantile(q=[0.01, 0.99])
    )

    nrows_prior = datasets.num_rows

    datasets = datasets.filter(
        lambda example: example["length_pre"] <= upper_length
        and example["length_pre"] >= lower_length
        and example["length_delta"] >= subtractive_bound
        and example["length_delta"] <= additive_bound
    )

    print(
        f"Number of records removed: { {k: nrows_prior[k] - v.num_rows for k, v in datasets.items()} }"
    )

    return datasets