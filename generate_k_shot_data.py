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
import re

from collections import defaultdict
from pandas import DataFrame

from templates import TEMPLATES

N_LABELS_DICT = {"SST-2": 2, "sst-5": 5, "mr": 2, "cr": 2, "mpqa": 2,
                 "subj": 2, "trec": 6, "CoLA": 2,
                 "amazon": 5, "yelp_full": 5, "yelp_binary": 2,
                 "agnews": 4, "copa": 2, "boolq": 2,
                 "RTE": 2, "cb": 3,
                 "yahoo": 10, "dbpedia": 14, 'climate_fever': 4, 
                 'ethos-national_origin': 2, 'ethos-race': 2,
                 'ethos-religion': 2, 'financial_phrasebank': 3, 
                 'hate_speech18': 2, 'medical_questions_pairs': 2, 
                 'poem_sentiment': 4, 'superglue-cb': 3, 
                 'tweet_eval-hate': 2, 'tweet_eval-stance_atheism': 3, 
                 'tweet_eval-stance_feminist': 3, 'anli': 3, 
                 'glue-mnli': 3, 'glue-qnli': 2, 'glue-rte': 2, 
                 'glue-wnli': 2, 'scitail': 2, 'sick': 3,
                 'ai2_arc': 4, 'codah': 4, 'commonsense_qa': 5, 
                 'openbookqa': 4, 'qasc': 8, 'quarel': 2, 'quartz-no_knowledge': 2, 
                 'quartz-with_knowledge': 2, 'superglue-copa': 2, 'wino_grande': 2
}


CROSSFIT_DATASETS = ['climate_fever', 'ethos-national_origin', 'ethos-race', 
                     'ethos-religion', 'financial_phrasebank', 'hate_speech18', 
                     'medical_questions_pairs', 'poem_sentiment', 'superglue-cb', 
                     'tweet_eval-hate', 'tweet_eval-stance_atheism', 'tweet_eval-stance_feminist',
                     'anli', 'glue-mnli', 'glue-qnli', 'glue-rte', 'glue-wnli', 'scitail', 'sick',
                     'ai2_arc', 'codah', 'commonsense_qa', 'cosmos_qa', 'dream', 'hellaswag', 
                     'openbookqa', 'qasc', 'quail', 'quarel', 'quartz-no_knowledge', 
                     'quartz-with_knowledge', 'race-high', 'race-middle', 'sciq', 'social_i_qa', 
                     'superglue-copa', 'swag', 'wino_grande', 'wiqa']

MC_DATASETS = ['ai2_arc', 'codah', 'commonsense_qa', 'cosmos_qa', 'dream', 'hellaswag', 
               'openbookqa', 'qasc', 'quail', 'quarel', 'quartz-no_knowledge', 
               'quartz-with_knowledge', 'race-high', 'race-middle', 'sciq', 'social_i_qa', 
               'superglue-copa', 'swag', 'wino_grande', 'wiqa']


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
    elif task in CROSSFIT_DATASETS:
        return line[-1]
    else:
        return line[0]


LABEL_WORDS = {
    'climate_fever': ["Supports", "Refutes", "Not enough info", "Disputed"],
    'ethos-national_origin': ["false", "true"],
    'ethos-race': ["false", "true"],
    'ethos-religion': ["false", "true"],
    'financial_phrasebank': ["positive", "negative", "neutral"],
    'hate_speech18': ["hate", "noHate"],
    'medical_questions_pairs': ["Similar", "Dissimilar"],
    'poem_sentiment': ["negative", "positive", "no_impact", "mixed"],
    'superglue-cb': ["entailment", "contradiction", "neutral"],
    'tweet_eval-hate': ["non-hate", "hate"],
    'tweet_eval-stance_atheism': ["none", "against", "favor"],
    'tweet_eval-stance_feminist': ["none", "against", "favor"],

    'anli': ["entailment", "neutral", "contradiction"],
    'glue-mnli': ["entailment", "contradiction", "neutral"], 
    'glue-qnli': ["entailment", "not_entailment"], 
    'glue-rte': ["entailment", "not_entailment"],
    'glue-wnli': ["entailment", "not_entailment"],
    'scitail': ["entailment", "neutral"], 
    'sick': ["entailment", "contradiction", "neutral"],
}


def format_sent_label(task, line, template_idx):
    line = line.strip().split('\t')
    sent = "\t".join(line[:-1])
    label = line[-1]
    
    if task in MC_DATASETS:
        sent_pieces = re.split('\([ABCDEFGH]\)', sent)
        if len(sent_pieces) != N_LABELS_DICT[task]+1:
            return None
        sent = sent_pieces[0].strip()
        choices = [sent_pieces[i].strip() for i in range(1, N_LABELS_DICT[task]+1)]
        try:
            label_id = [i for i, piece in enumerate(choices) if piece == label][0]
        except:
            return None
        choices_string = '!@#'.join(choices)
        sent = TEMPLATES[task][template_idx][0] % (sent)

        return sent + '\t' + str(label_id) + '\t' + choices_string + '\n'
    else:
        if task in ["medical_questions_pairs", "superglue-cb", "anli", "glue-mnli", "glue-qnli", "glue-rte", "glue-wnli", "scitail", "sick"]:
            sentences = sent.split('[SEP]')
            sent_pieces = [sentence[sentence.index(':')+1 : ].strip() for sentence in sentences]
            sent = TEMPLATES[task][template_idx][0] % (tuple(sent_pieces))
        else:    
            sent = TEMPLATES[task][template_idx][0] % (sent)
        label_id = LABEL_WORDS[task].index(label)

        return sent + '\t' + str(label_id) + '\n'


def load_datasets(data_dir, tasks, k=-1):
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
        elif task in CROSSFIT_DATASETS:
            
            n_templates = 1

            # CrossFit
            dataset = {}
            dirname = os.path.join(data_dir, task)
            if k != 16384:
                splits = ["train", "dev"]
                for split in splits:
                    # combine files
                    lines = [[] for i in range(n_templates)]
                    for s in [13, 21, 42, 87, 100]:
                        filename = os.path.join(dirname, "{}_16_{}_{}.tsv".format(task, s, split))
                        with open(filename, "r") as f:
                            for line in f:
                                for template_idx in range(n_templates):
                                    formatted = format_sent_label(task, line, template_idx)
                                    if formatted != None:
                                        lines[template_idx].append(formatted)
                    dataset[split] = lines
                filename = os.path.join(dirname, "{}_16384_100_test.tsv".format(task))
                lines = [[] for i in range(n_templates)]
                with open(filename, "r") as f:
                    for line in f:
                        for template_idx in range(n_templates):
                            formatted = format_sent_label(task, line, template_idx)
                            if formatted != None:
                                lines[template_idx].append(formatted)
                dataset["test"] = lines
            else:
                splits = ["train", "dev", "test"]
                for split in splits:
                    filename = os.path.join(dirname, "{}_16384_100_{}.tsv".format(task, split))
                    lines = [[] for i in range(n_templates)]
                    with open(filename, "r") as f:
                        for line in f:
                            for template_idx in range(n_templates):
                                formatted = format_sent_label(task, line, template_idx)
                                if formatted != None:
                                    lines[template_idx].append(formatted)

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
                 'agnews', 'amazon', 'yelp_full', 'dbpedia', 'yahoo',
                 'climate_fever', 'ethos-national_origin', 'ethos-race', 
                 'ethos-religion', 'financial_phrasebank', 'hate_speech18',
                 'medical_questions_pairs', 'poem_sentiment',
                 'superglue-cb', 'tweet_eval-hate', 
                 'tweet_eval-stance_atheism', 'tweet_eval-stance_feminist',
                 'anli', 'glue-mnli', 'glue-qnli', 'glue-rte', 'glue-wnli', 'scitail', 'sick',
                 'ai2_arc', 'codah', 'commonsense_qa', 'cosmos_qa', 'dream', 'hellaswag', 
                 'openbookqa', 'qasc', 'quail', 'quarel', 'quartz-no_knowledge', 
                 'quartz-with_knowledge', 'race-high', 'race-middle', 'sciq', 
                 'social_i_qa', 'superglue-copa', 'swag', 'wino_grande', 'wiqa'], #, 'mpqa', 'CoLA', 'MRPC', 'QQP', 'STS-B', 'MNLI', 'SNLI', 'QNLI', 'RTE'],
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
    main_for_crossfit(args, [task for task in args.task
                          if task in CROSSFIT_DATASETS])

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
                    if k == -1:
                        for label in label_list:
                            for line in label_list[label]:
                                f.write(line)
                    else:
                        for label in label_list:
                            for line in label_list[label][:k*n_classes]:
                                f.write(line)
                name = "dev.tsv"
                if task == 'MNLI':
                    name = "dev_matched.tsv"
                with open(os.path.join(setting_dir, name), "w") as f:
                    for line in train_header:
                        f.write(line)
                    if k == -1:
                        for label in label_list:
                            for line in label_list[label]:
                                f.write(line)
                    else:
                        for label in label_list:
                            dev_rate = 11 if '10x' in args.mode else 2
                            for line in label_list[label][k:k*dev_rate]:
                                f.write(line)
            else:
                new_train = []
                if k == -1:
                    for label in label_list:
                        for line in label_list[label]:
                            new_train.append(line)
                else:
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


def main_for_crossfit(args, tasks):
    k = args.k
    datasets = load_datasets(os.path.join(args.data_dir, "CrossFitDatasets"), tasks, k=k)

    for seed in args.seed:
        for task, dataset in datasets.items():
            # Set random seed
            np.random.seed(seed)

            # Set up dir
            task_dir = os.path.join(args.output_dir, task)
            setting_dir = os.path.join(task_dir, "{}-{}".format(k, seed))
            os.makedirs(setting_dir, exist_ok=True)

            for template_idx in range(len(dataset["train"])):
                # Set up dir
                template_dir = os.path.join(setting_dir, "{}".format(template_idx))
                os.makedirs(template_dir, exist_ok=True)

                if k != 16384:
                    # Write test splits
                    with open(os.path.join(template_dir, "test.tsv"), "w") as f:
                        for line in dataset["test"][template_idx]:
                            f.write(line)

                    # Shuffle the training set
                    train_lines = dataset['train'][template_idx]
                    np.random.shuffle(train_lines)

                    # Get label list for balanced sampling
                    label_list = {}
                    for line in train_lines:
                        label = get_label(task, line)
                        if not args.balance:
                            label = "all"
                        if label not in label_list:
                            label_list[label] = [line]
                        else:
                            label_list[label].append(line)
                    n_classes = 1 #if args.balance else len(label_set)

                    # Write the training split
                    with open(os.path.join(template_dir, "train.tsv"), "w") as f:
                        for label in label_list:
                            for line in label_list[label][:k*n_classes]:
                                f.write(line)
                    
                    # Write the development split
                    with open(os.path.join(template_dir, "dev.tsv"), "w") as f:
                        for label in label_list:
                            dev_rate = 11 if '10x' in args.mode else 2
                            for line in label_list[label][k:k*dev_rate]:
                                f.write(line)

                else:
                    # Write test splits
                    with open(os.path.join(template_dir, "test.tsv"), "w") as f:
                        for line in dataset["test"][template_idx]:
                            f.write(line)

                    # Write the training split
                    with open(os.path.join(template_dir, "train.tsv"), "w") as f:
                        for line in dataset["train"][template_idx]:
                            f.write(line)

                    # Write the training split
                    with open(os.path.join(template_dir, "dev.tsv"), "w") as f:
                        for line in dataset["dev"][template_idx]:
                            f.write(line)

            if seed==args.seed[-1]:
                print ("Done for task=%s" % task)


if __name__ == "__main__":
    main()
