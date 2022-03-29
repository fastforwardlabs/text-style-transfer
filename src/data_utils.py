import os
from functools import reduce
from operator import concat
from collections import defaultdict

import pandas as pd
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict, Value, Translation, Features


def build_hf_dataset(path: str, one_word=True) -> DatasetDict:
    """
    Formats the raw data into HuggingFace DatasetDict object.

    If one_word = True:
        Provided a path to the raw Wiki Neutrality Corpus (WNC) data files,
        this function parses each of the dev, test, and train sets the "one-word" version
        and formats them as a HuggingFace DatasetDict object in the seq2seq style
        (i.e. ready for translation tasks).

    If one_word = False:
        Provided a path to the raw Wiki Neutrality Corpus (WNC) data files,
        this function parses the full version of the dataset and formats it as a HuggingFace
        DatasetDict object in the seq2seq style (i.e. ready for translation tasks) and splits
        into train/dev/test sets (90/5/5).

    https://arxiv.org/pdf/1911.09709.pdf

    Args:
        path (str): path to directory containing raw WNC data files
        one_word (str): prepare one word edit version or full version of dataset

    Returns:
        DatasetDict

    """

    if one_word:
        splits = ["dev", "test", "train"]
        base_path = "biased.word."
    else:
        splits = ["full"]
        base_path = "biased."

    dataset_dict = defaultdict(dict)

    FEATURES = Features(
        {
            "rev_id": Value("string"),
            "translation": Translation(languages=["pre", "post"]),
        }
    )

    for split in splits:

        PATH = os.path.join(path, base_path + split)

        rev_ids = []
        translation_pairs = []

        for i, line in enumerate(tqdm(open(PATH))):
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

    dataset_dict = DatasetDict(dataset_dict)

    if not one_word:
        dataset_dict = create_dataset_splits(dataset_dict)

    return dataset_dict


def create_dataset_splits(wnc_datasets_full: DatasetDict) -> DatasetDict:
    """
    Provided a DatasetDict containing one key for the full WNC datasets, this
    function breaks the full dataset into three splits train, dev, test.

    """
    # split 90% train / 10% test
    wnc_datasets_full = wnc_datasets_full["full"].train_test_split(
        train_size=0.9, shuffle=True, seed=42
    )

    # split 10% into 5% dev / 5% test
    train_test_temp = wnc_datasets_full["test"].train_test_split(
        train_size=0.5, shuffle=True, seed=42
    )

    # cleanup and curate DatsetDict object
    wnc_datasets_full.pop("test")
    wnc_datasets_full["test"] = train_test_temp["test"]
    wnc_datasets_full["dev"] = train_test_temp["train"]
    del train_test_temp

    return wnc_datasets_full


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
    rev_ids_pd = pd.Series(rev_ids)
    duplicate_revids = rev_ids_pd[rev_ids_pd.duplicated()].tolist()

    print(f"{len(duplicate_revids)*2} duplicate records have been removed.")

    return datasets.filter(lambda x: x["rev_id"] not in duplicate_revids)


def remove_outliers(datasets: DatasetDict, one_word=True) -> DatasetDict:
    """
    Remove outlier observations from the datasets.

    Provided a DatasetDict of the WNC, this function filters out all outlier
    records as defined by:

    If one_word = True:
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

    If one_word = False:
        1. Records with pre-edit sentence length > 99th percentile.
        2. Records with pre-edit sentence length < 1st percentile.
        3. Records with net subtraction of words (length_delta) < 1st percentile.
        4. Records with net addition of words (length_delta) > 99th percentile.

    Args:
        datasets (DatasetDict): a WNC datasets object returned by `build_hf_dataset()`
        one_word (str): prepare one word edit version or full version of dataset

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
    # NOTE - this differs between versions of the dataset
    if one_word:
        subtractive_bound = -1
        additive_bound = 4
    else:
        subtractive_bound, additive_bound = (
            datasets["train"].to_pandas()["length_delta"].quantile(q=[0.01, 0.99])
        )

    lower_length, upper_length = (
        datasets["train"].to_pandas()["length_pre"].quantile(q=[0.01, 0.99])
    )

    print(f"Lower length bound: {lower_length} | Upper length bound: {upper_length}")
    print(
        f"Lower length_delta bound: {subtractive_bound} | Upper length_delta bound: {additive_bound}"
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


def plot_length_dist_bysplit(datasets):
    """
    Plot distribution of text lengths for both pre and post edits across
    each of the three splits

    """

    for split in datasets.keys():
        df = datasets[split].to_pandas()
        sns.displot(df[["length_pre", "length_post"]], kind="hist", aspect=2).set(
            title=f"{split.capitalize()} Split -- Distribution of Text Length Pre/Post Edit"
        )


def get_descriptive_stats(datasets):
    """
    Get descriptive statistics for each split in the datasets object across
    both pre and post edits. Format as a dataframe and return.

    """

    # collect descriptive statistics for each split
    stats = {}
    for split in datasets.keys():
        df = datasets[split].to_pandas()
        stat = df[["length_pre", "length_post"]].describe().round(2)
        stats[split] = stat

    # format as dataframe
    return pd.concat(stats).unstack(0).swaplevel(0, 1, axis=1).sort_index(axis=1)


def plot_length_delta_bysplit(datasets):
    """
    Plot the distribution of change in text length after edits by dataset split.

    """

    # aggregate number of revision by the change in length
    deltas = {}
    for split in datasets.keys():
        delta = (
            datasets[split]
            .to_pandas()[["rev_id", "length_delta"]]
            .groupby("length_delta")
            .count()
        )
        deltas[split] = delta

    # format for plotting
    delta_df = pd.concat(deltas).unstack(0).droplevel(0, axis=1).fillna(0)

    for col in delta_df.columns:
        delta_df[col] = delta_df[col] / delta_df[col].sum()

    delta_df.plot.bar(
        title="Percentage of Edits by Change in Word Count (by Split)", figsize=(17, 8)
    )
