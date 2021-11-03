import os
import csv
import json

import numpy as np
import torch

from util import prepro_sentence, prepro_sentence_pair, \
    prepro_sentence_pair_single

def load_data(data_dir, task, k, seed, split, template_idx=None):
    data_dir = os.path.join(data_dir, "k-shot", task, "{}-{}".format(k, seed))
    if template_idx is not None:
        data_dir = os.path.join(data_dir, str(template_idx))
    data = []
    if os.path.exists(os.path.join(data_dir, "{}.tsv".format(split))):
        with open(os.path.join(data_dir, "{}.tsv".format(split)), "r") as f:
            for line in f:
                data.append(line.strip().replace("\\n", '\n').split("\t"))
        if task=="CoLA":
            data = [(sent, label) for _, label, _, sent in data]
        elif task=="RTE":
            data = [(json.dumps({
                "text": p, "question": h[:-1] if h.endswith(".") else h
            }), "1" if l=="entailment" else "0")
                    for _, p, h, l in data[1:]]
        elif data[0]==["sentence", "label"]:
            data = data[1:]
    elif os.path.exists(os.path.join(data_dir, "{}.csv".format(split))):
        with open(os.path.join(data_dir, "{}.csv".format(split)), "r") as f:
            for label, text in csv.reader(f):
                data.append((text, label))
    else:
        raise NotImplementedError(data_dir)

    # # all data should have (input, output) format
    # assert np.all([len(dp)==2 for dp in data])

    return data


def prepare_data(tokenizer, train_data, test_data, max_length, max_length_per_example,
                 n_classes=2, templates=None, method_type="generative",
                 is_training=False, use_demonstrations=False,
                 ensemble=False, is_null=False):
    is_multiple_choice = False
    if templates==None:
        transform = None
        is_multiple_choice = True
        templates = []
        for _, _, choices_string in test_data:
            choices = choices_string.split("!@#")
            assert(len(choices)==n_classes)
            templates += choices
        test_data = [(sent, label) for sent, label, _ in test_data]
        if train_data != None:
            train_data = [(sent, label) for sent, label, _ in train_data]
    elif type(templates)==list:
        transform = None
        assert len(templates)==n_classes
    else:
        transform = templates
    assert method_type in ["direct", "channel"]

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    '''
    if method==direct, "sent prompt sent prompt ..."
    - prompt should have space
    - if demonstrations are used, 2nd sentneces to the input sentence should have space

    if method==channel, "prompt sent prompt sent ..."
    - input sent should have space
    - if demonstrations are used, 2nd prompts to the input prompt should have space
    '''

    # For calibration method, following Zhao et al. 2021
    if is_null:
        assert test_data is None
        assert method_type=="direct"
        test_data = [("N/A", "0")]

    prefixes_with_space = None
    if transform is None:
        templates = [template.strip() for template in templates]
        if method_type=="direct":
            templates = [" "+template for template in templates]
            if use_demonstrations:
                test_data = [(" "+sent, label) for sent, label in test_data]
        elif method_type=="channel":
            test_data = [(" "+sent, label) for sent, label in test_data]
            if train_data is not None:
                train_data = [(" "+sent, label) for sent, label in train_data]
            prefixes_with_space = [tokenizer(" "+template)["input_ids"] for template in templates]
        else:
            raise NotImplementedError()

    if transform is None:
        test_inputs = [tokenizer(sent)["input_ids"] for sent, _ in test_data]
        truncated = np.sum([len(inputs)>max_length_per_example-16 for inputs in test_inputs])

        if truncated > 0:
            test_inputs = [inputs[:max_length_per_example-16] for inputs in test_inputs]
            print ("%d/%d truncated" % (truncated, len(test_inputs)))

        prefixes = [tokenizer(template)["input_ids"] for template in templates]
        idx = [idx for idx, _prefixes in enumerate(zip(*prefixes))
                if not np.all([_prefixes[0]==_prefix for _prefix in _prefixes])][0]


    else:
        test_inputs = [transform(dp, tokenizer,
                                 max_length_per_example-16,
                                 groundtruth_only=is_training)
                                   for dp in test_data]
        if not is_training:
            assert np.all([len(dp)==2 and
                           np.all([len(dpi)==n_classes for dpi in dp])
                           for dp in test_inputs])


    if is_training:
        assert not use_demonstrations
        assert not ensemble

        input_ids, attention_mask, token_type_ids, classes = [], [], [], []
        for i, (test_input, dp) in enumerate(zip(test_inputs, test_data)):
            if transform is not None:
                test_input, test_output = test_input
                encoded = prepro_sentence_pair_single(
                    test_input, test_output, max_length, bos_token_id, eos_token_id
                )
            else:
                prefix = prefixes[int(dp[1])] if not is_multiple_choice else prefixes[i * n_classes + int(dp[1])]
                if method_type=="channel":
                    encoded = prepro_sentence_pair_single(
                        prefix, test_input, max_length, bos_token_id, eos_token_id)
                elif method_type=="direct":
                    encoded = prepro_sentence_pair_single(
                        test_input + prefix[:idx], prefix[idx:], max_length, bos_token_id, eos_token_id)
                else:
                    raise NotImplementedError()
            input_ids.append(encoded[0])
            attention_mask.append(encoded[1])
            token_type_ids.append(encoded[2])
            classes.append(int(dp[1]))
        return dict(input_ids=torch.LongTensor(input_ids),
                    attention_mask=torch.LongTensor(attention_mask),
                    token_type_ids=torch.LongTensor(token_type_ids),
                    classes=torch.LongTensor(classes))

    if use_demonstrations:

        if transform is not None:
            raise NotImplementedError()

        if ensemble:
            return prepare_data_for_parallel(
                tokenizer, train_data, test_data,
                max_length, max_length_per_example,
                method_type, n_classes,
                test_inputs, prefixes, idx, prefixes_with_space,
                bos_token_id, eos_token_id)


        assert train_data is not None
        demonstrations = []

        np.random.shuffle(train_data)

        for sent, label in train_data:
            if len(demonstrations)>0:
                if method_type=="direct":
                    sent = " " + sent
                elif method_type=="channel":
                    prefixes = prefixes_with_space

            if transform is None:
                tokens = tokenizer(sent)["input_ids"][:max_length_per_example]
            else:
                tokens = transform(sent, tokenizer, max_length_per_example)
            prefix = prefixes[(int(label))]

            if method_type=="channel":
                tokens = prefix + tokens
            elif method_type=="direct":
                tokens = tokens + prefix
            else:
                raise NotImplementedError()

            demonstrations += tokens

    if transform is None:
        # check if idx is set well
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                assert prefixes[i][:idx]==prefixes[j][:idx]
                assert prefixes[i][idx]!=prefixes[j][idx]

    input_tensors = []

    for i in range(n_classes):
        if is_multiple_choice:
            input_ids, attention_mask, token_type_ids = [], [], []
            for j, test_input in enumerate(test_inputs):
                prefix = prefixes[j * n_classes + i].copy()
                if method_type=="channel":
                    encoded = prepro_sentence_pair_single(
                        prefix, test_input, max_length, bos_token_id, eos_token_id)
                elif method_type=="direct":
                    encoded = prepro_sentence_pair_single(
                        test_input + prefix[:idx], prefix[idx:], max_length, bos_token_id, eos_token_id)
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])
            tensor = dict(input_ids=torch.LongTensor(input_ids),
                          attention_mask=torch.LongTensor(attention_mask),
                          token_type_ids=torch.LongTensor(token_type_ids))
        elif transform is None:
            prefix = prefixes[i].copy()
            if method_type=="channel":
                if use_demonstrations:
                    prefix = demonstrations.copy() + prefix
                tensor = prepro_sentence_pair([prefix], test_inputs, max_length,
                                            bos_token_id, eos_token_id,
                                            allow_truncation=use_demonstrations)
            elif method_type=="direct":
                if use_demonstrations:
                    prompt = [demonstrations.copy() + test_input + prefix[:idx] for test_input in test_inputs]
                else:
                    prompt = [test_input + prefix[:idx] for test_input in test_inputs]
                tensor = prepro_sentence_pair(prompt,
                                            [prefix[idx:]], max_length,
                                            bos_token_id, eos_token_id,
                                            allow_truncation=use_demonstrations)
            else:
                raise NotImplementedError()
        else:
            input_ids, attention_mask, token_type_ids = [], [], []
            for input_, output_ in test_inputs:
                encoded = prepro_sentence_pair_single(
                    input_[i], output_[i], max_length,
                    bos_token_id,
                    None if is_generation else eos_token_id,
                    allow_truncation=False)
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])
            tensor = dict(input_ids=torch.LongTensor(input_ids),
                          attention_mask=torch.LongTensor(attention_mask),
                          token_type_ids=torch.LongTensor(token_type_ids))

        tensor["classes"] = torch.LongTensor([i] * len(tensor["input_ids"]))
        input_tensors.append(tensor)


    return input_tensors


def prepare_data_for_parallel(tokenizer, train_data, test_data,
                              max_length, max_length_per_example,
                              method_type, n_classes,
                              test_inputs, prefixes, idx, prefixes_with_space,
                              bos_token_id, eos_token_id):

    # get len(train_data) number of demonstrations

    assert train_data is not None
    demonstrations_list = []

    np.random.shuffle(train_data)

    for sent, label in train_data:
        tokens = tokenizer(sent)["input_ids"][:max_length_per_example]
        prefix = prefixes[(int(label))]
        if method_type=="channel":
            tokens = prefix + tokens
        elif method_type=="direct":
            tokens = tokens + prefix
        else:
            raise NotImplementedError()

        demonstrations_list.append(tokens)

    # check if idx is set well
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            assert prefixes[i][:idx]==prefixes[j][:idx]
            assert prefixes[i][idx]!=prefixes[j][idx]

    input_tensors = []

    for i in range(n_classes):

        if method_type=="channel":
            prefix = prefixes_with_space[i].copy()
            prompt = [demonstrations + prefix
                      for demonstrations in demonstrations_list]
            tensor = prepro_sentence_pair(
                prompt, test_inputs, max_length,
                bos_token_id, eos_token_id,
                allow_truncation=True)

        elif method_type=="direct":
            prefix = prefixes[i].copy()
            prompt = [demonstrations.copy() + test_input + prefix[:idx]
                      for test_input in test_inputs
                      for demonstrations in demonstrations_list]

            tensor = prepro_sentence_pair(prompt,
                                          [prefix[idx:]], max_length,
                                          bos_token_id, eos_token_id,
                                          allow_truncation=True)
        else:
            raise NotImplementedError()

        input_tensors.append(tensor)


    return input_tensors

def load_prompt(prompts_dir, prompt_task):
    prompt_files = ["channel_prompts", "natural_prompts", "pile"]
    prompts = {}
    for prompt_file in prompt_files:
        with open(os.path.join(prompts_dir, prompt_file+".json"), 'r') as f:
            prompts.update(json.load(f))
    if prompt_task not in prompts:
        raise NotImplementedError()
    return prompts[prompt_task]


def output_metrices(args, results, prompt, best_lr):
    metrices = {
        "taskA": args.task,
        "taskB": args.prompt_task,
        "discrete_prompt": prompt,
        "optimize_against_A": args.bad,
        "gamma": args.aux_weight,
        "learning_rate": best_lr,
        "average_accuracy_A": np.mean([result[0] for result in results]),
        "worst_accuracy_A": np.min([result[0] for result in results])
    }

    with open(os.path.join(args.out_dir, "metrics.json"), 'w') as f:
        json.dump(metrices, f)
    