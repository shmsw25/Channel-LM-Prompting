"""
This is a modification of the preprocessing code from Gao et al. ACL 2021, "Making Pre-trained Language Models Better Few-shot Learners".
Paper: https://arxiv.org/abs/2012.15723
Original script: https://github.com/princeton-nlp/LM-BFF/blob/main/tools/generate_k_shot_data.py
The code was modified to add options for
(1) not guaranteeing the balance between labels in the training data,
(2) k training examples in total instead of per label, and
(3) add datasets from Zhang et al.
"""

"""This script samples K examples randomly without replacement from the original data."""

import argparse
import os
import csv
import numpy as np
import pandas as pd

from collections import defaultdict
from pandas import DataFrame

def get_label(task, line):
    if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
        # GLUE style
        line = line.strip().split('\t')
        if task == 'CoLA':
            return line[1]
        elif task == 'MNLI':
            return line[-1]
        elif task == 'MRPC':
            return line[0]
        elif task == 'QNLI':
            return line[-1]
        elif task == 'QQP':
            return line[-1]
        elif task == 'RTE':
            return line[-1]
        elif task == 'SNLI':
            return line[-1]
        elif task == 'SST-2':
            return line[-1]
        elif task == 'STS-B':
            return 0 if float(line[-1]) < 2.5 else 1
        elif task == 'WNLI':
            return line[-1]
        else:
            raise NotImplementedError
    else:
        return line[0]

def load_datasets(data_dir, tasks):
    datasets = {}
    for task in tasks:
        if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
            # GLUE style (tsv)
            dataset = {}
            dirname = os.path.join(data_dir, task)
            if task == "MNLI":
                splits = ["train", "dev_matched", "dev_mismatched"]
            else:
                splits = ["train", "dev"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.tsv")
                with open(filename, "r") as f:
                    lines = f.readlines()
                dataset[split] = lines
            datasets[task] = dataset
        else:
            # Other datasets (csv)
            dataset = {}
            dirname = os.path.join(data_dir, task)
            splits = ["train", "test"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.csv")
                dataset[split] = pd.read_csv(filename, header=None)
            datasets[task] = dataset
    return datasets

def split_header(task, lines):
    """
    Returns if the task file has a header or not. Only for GLUE tasks.
    """
    if task in ["CoLA"]:
        return [], lines
    elif task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI"]:
        return lines[0:1], lines[1:]
    else:
        raise ValueError("Unknown GLUE task.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16,
        help="Training examples for each class.")
    parser.add_argument("--task", type=str, nargs="+",
        default=['SST-2', 'sst-5', 'mr', 'cr', 'subj', 'trec',
                 'agnews', 'amazon', 'yelp_full', 'dbpedia', 'yahoo'], #, 'mpqa', 'CoLA', 'MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE'],
        help="Task names")
    parser.add_argument("--seed", type=int, nargs="+",
        default=[100, 13, 21, 42, 87],
        help="Random seeds")
    parser.add_argument("--data_dir", type=str, default="data/original", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output path")
    parser.add_argument("--mode", type=str, default='k-shot', choices=['k-shot', 'k-shot-10x'], help="k-shot or k-shot-10x (10x dev set)")
    parser.add_argument("--balance", action="store_true")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.mode)

    main_for_gao(args, [task for task in args.task
                        if task in ["SST-2", "sst-5", "mr", "cr", "trec", "subj"]])
    main_for_zhang(args, [task for task in args.task
                          if task in ["agnews", "amazon", "yelp_full", "dbpedia", "yahoo"]])

def main_for_gao(args, tasks):
    k = args.k
    #print("K =", k)
    datasets = load_datasets(args.data_dir, tasks)

    for seed in args.seed:
        #print("Seed = %d" % (seed))
        for task, dataset in datasets.items():
            # Set random seed
            np.random.seed(seed)

            # Shuffle the training set
            #print("| Task = %s" % (task))
            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                # GLUE style
                train_header, train_lines = split_header(task, dataset["train"])
                np.random.shuffle(train_lines)
            else:
                # Other datasets
                train_lines = dataset['train'].values.tolist()
                np.random.shuffle(train_lines)

            # Set up dir
            task_dir = os.path.join(args.output_dir, task)
            setting_dir = os.path.join(task_dir, "{}-{}".format(k, seed))
            os.makedirs(setting_dir, exist_ok=True)

            # Get label list for balanced sampling
            label_list = {}
            label_set = set()

            for line in train_lines:
                label = get_label(task, line)
                label_set.add(label)

            for line in train_lines:
                label = get_label(task, line)
                if not args.balance:
                    label = "all"
                if label not in label_list:
                    label_list[label] = [line]
                else:
                    label_list[label].append(line)

            n_classes = 1 #if args.balance else len(label_set)

            # Write test splits
            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                # GLUE style
                # Use the original development set as the test set (the original test sets are not publicly available)
                for split, lines in dataset.items():
                    if split.startswith("train"):
                        continue
                    lines = dataset[split]
                    split = split.replace('dev', 'test')
                    with open(os.path.join(setting_dir, f"{split}.tsv"), "w") as f:
                        for line in lines:
                            f.write(line)


            else:
                # Other datasets
                # Use the original test sets
                dataset['test'].to_csv(os.path.join(setting_dir, 'test.csv'), header=False, index=False)

            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                with open(os.path.join(setting_dir, "train.tsv"), "w") as f:
                    for line in train_header:
                        f.write(line)
                    for label in label_list:
                        for line in label_list[label][:k*n_classes]:
                            f.write(line)
                name = "dev.tsv"
                if task == 'MNLI':
                    name = "dev_matched.tsv"
                with open(os.path.join(setting_dir, name), "w") as f:
                    for line in train_header:
                        f.write(line)
                    for label in label_list:
                        dev_rate = 11 if '10x' in args.mode else 2
                        for line in label_list[label][k:k*dev_rate]:
                            f.write(line)
            else:
                new_train = []
                for label in label_list:
                    for line in label_list[label][:k*n_classes]:
                        new_train.append(line)
                new_train = DataFrame(new_train)
                new_train.to_csv(os.path.join(setting_dir, 'train.csv'), header=False, index=False)

                new_dev = []
                for label in label_list:
                    dev_rate = 11 if '10x' in args.mode else 2
                    for line in label_list[label][k:k*dev_rate]:
                        new_dev.append(line)
                new_dev = DataFrame(new_dev)
                new_dev.to_csv(os.path.join(setting_dir, 'dev.csv'), header=False, index=False)

            #print (setting_dir)

            if seed==args.seed[-1]:
                print ("Done for task=%s" % task)

DATA_DICT = {"agnews": "ag_news_csv",
             "amazon": "amazon_review_full_csv",
             "dbpedia": "dbpedia_csv",
             "yahoo": "yahoo_answers_csv",
             "yelp_full": "yelp_review_full_csv"}

def main_for_zhang(args, tasks):
    for task in tasks:
        for seed in args.seed:
            for split in ["train", "test"]:
                prepro_for_zhang(task, split, seed, args)
        print ("Done for task=%s" % task)

def prepro_for_zhang(dataname, split, seed, args):
    balance = args.balance
    k = args.k
    np.random.seed(seed)

    data = defaultdict(list)
    label_set = set()
    with open(os.path.join(args.data_dir, "TextClassificationDatasets", DATA_DICT[dataname], "{}.csv".format(split)), "r") as f:
        for dp in csv.reader(f, delimiter=","):
            if "yelp" in dataname:
                assert len(dp)==2
                label, sent = dp[0], dp[1]
            elif "yahoo" in dataname:
                assert len(dp)==4
                label = dp[0]
                dp[3] = dp[3].replace("\t", " ").replace("\\n", " ")
                sent = " ".join(dp[1:])
            else:
                assert len(dp)==3
                if "\t" in dp[2]:
                    dp[2] = dp[2].replace("\t", " ")
                label, sent = dp[0], dp[1] + " " + dp[2]
            label = str(int(label)-1)
            label_set.add(label)
            if balance:
                data[label].append((sent, label))
            else:
                data["all"].append((sent, label))

    n_classes = len(label_set)
    save_base_dir = os.path.join(args.output_dir, dataname)

    if not os.path.exists(save_base_dir):
        os.mkdir(save_base_dir)

    labels = set(list(data.keys()))

    if split!="test":
        for label in data:
            np.random.shuffle(data[label])

    save_dir = os.path.join(save_base_dir, "{}-{}".format(k, seed))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, "{}.tsv".format(split))
    with open(save_path, "w") as f:
        f.write("sentence\tlabel\n")
        for sents in data.values():
            if split=="test":
                pass
            elif balance:
                sents = sents[:k]
            else:
                sents = sents[:k]
            for sent, label in sents:
                assert "\t" not in sent, sent
                f.write("%s\t%s\n" % (sent, label))

if __name__ == "__main__":
    main()
