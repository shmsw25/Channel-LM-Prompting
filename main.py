import os
import argparse
import pickle as pkl
import random
import torch
import math
import logging
import numpy as np

from tqdm import tqdm
from collections import Counter, defaultdict

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.deepspeed import HfDeepSpeedConfig

from data import load_data, prepare_data
from run import train, inference
from model_util import load_checkpoint, set_extra_embeddings, \
    set_separate_lm_head, set_separate_embeddings, set_transformed_lm_head, set_prior
from util import get_prompts, get_paths, flatten_label_losses, \
    prepend_task_tokens, reassign_output_tokens

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
                 'tweet_eval-stance_feminist': 3
}


def main(logger, args):
    args.gpt2 = args.gpt2.replace("gpt2-small", "gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
    model = None

    if args.train_task is None:
        # standard case where the training task and the test task are the same
        train_task = args.task
    else:
        # zero-shot transfer case where the training task is different from the test task
        train_task = args.train_task
        assert args.do_check

    # datasets that also formats input sentence
    crossfit_datasets = ['climate_fever', 'ethos-national_origin',
                      'ethos-race', 'ethos-religion', 'financial_phrasebank',
                      'hate_speech18', 'medical_questions_pairs', 'poem_sentiment', 
                      'superglue-cb', 'tweet_eval-hate', 'tweet_eval-stance_atheism', 
                      'tweet_eval-stance_feminist']

    # datasets where the average input length is long
    long_datasets = ["cr", "subj", "agnews",
                     "amazon", "yelp_full", "yelp_binary", "boolq",
                     "dbpedia", "yahoo", 'climate_fever', 
                     'ethos-national_origin', 'ethos-race', 
                     'ethos-religion', 'financial_phrasebank', 'hate_speech18', 
                     'medical_questions_pairs', 'superglue-cb', 
                     'tweet_eval-hate' ]
    max_length = 256 if train_task in long_datasets else 128
    batch_size = int(args.batch_size / 2) if train_task in long_datasets else args.batch_size

    logger.info("%s %s" % (args.method, args.task))

    assert args.method in ["direct", "channel"]

    if args.use_demonstrations:
        assert args.do_zeroshot and not args.do_train

    if args.ensemble:
        assert args.use_demonstrations

    if args.do_train or args.use_demonstrations:
        assert args.train_seed > 0

    n_templates = 1
    k = int(args.k)
    seed = int(args.seed)
    local_rank = int(args.local_rank) if args.local_rank is not None else -1

    if args.task in crossfit_datasets:
        train_datas = [load_data(args.data_dir, train_task, k, seed, "train", template_idx=template_idx)
                       for template_idx in range(n_templates)] 
    else:
        train_data = load_data(args.data_dir, train_task, k, seed, "train")
    if args.split is None:
        assert args.do_zeroshot
        dev_data = None
    else:
        if args.task in crossfit_datasets:
            dev_datas = [load_data(args.data_dir, args.task, k, seed, args.split, template_idx=template_idx)
                         for template_idx in range(n_templates)]
        else:
            dev_data = load_data(args.data_dir, args.task, k, seed, args.split)

    if args.deep_speed:
        ds_config = {
            "train_batch_size": batch_size,
            "train_micro_batch_size_per_gpu": int(batch_size / torch.cuda.device_count()),
            "fp16": {
                "enabled": False,
                "loss_scale": 0,
                "initial_scale_power": 32,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                }
            },
            "zero_allow_untested_optimizer": True
        }
    else:
        ds_config = None

    accs, f1s = [], []
    # run over different templates
    for template_idx in range(n_templates):
        if args.task in crossfit_datasets:
            train_data = train_datas[template_idx]
            dev_data = dev_datas[template_idx] if args.split is not None else None
        acc, f1 = run(logger, args.do_train, args.do_zeroshot,
                  args.task, train_task,
                  k, seed, args.train_seed,
                  args.out_dir, args.split,
                  tokenizer, model, train_data, dev_data,
                  batch_size, max_length, args.gpt2,
                  template_idx, args.method,
                  args.lr, args.prior_weight, args.regularization_weight,
                  args.warmup_steps, ds_config, local_rank,
                  use_demonstrations=args.use_demonstrations,
                  use_calibration=args.use_calibration,
                  ensemble=args.ensemble,
                  is_null=args.split is None,
                  prompt_tune=args.prompt_tune,
                  head_tune=args.head_tune,
                  transform_tune=args.transform_tune,
                  prior_tune=args.prior_tune,
                  do_check=args.do_check,
                  n_prefix=args.n_prefix,
                  deep_speed=args.deep_speed)

        accs.append(acc)
        f1s.append(f1)

    if args.split is not None:
        logger.info("Accuracy = %.1f (Avg) / %.1f (Worst)" % (100*np.mean(accs), 100*np.min(accs)))
        logger.info("Micro-F1 = %.1f (Avg) / %.1f (Worst)" % (100*np.mean(f1s), 100*np.min(f1s)))


def run(logger, do_train, do_zeroshot, task, train_task, k, seed,
        train_seed,
        out_dir, split, tokenizer, model,
        train_data, dev_data,
        batch_size, max_length, gpt2, template_idx, method_type,
        learning_rate, prior_weight, regularization_weight,
        warmup_steps, ds_config, local_rank,
        use_demonstrations=False,
        use_calibration=False,
        ensemble=False,
        is_null=False,
        prompt_tune=False,
        head_tune=False,
        transform_tune=False,
        prior_tune=False,
        do_check=False, n_prefix=20,
        deep_speed=False):

    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)

    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(train_seed)

    if head_tune or transform_tune:
        assert method_type == "direct"

    n_classes = N_LABELS_DICT.get(task, None)
    templates = get_prompts(task, template_idx)

    n_classes_train = N_LABELS_DICT.get(train_task, None)
    templates_train = get_prompts(train_task, template_idx)

    # TODO: maybe need to consider this for the newly added datasets
    if task in ["yelp_full", "amazon"] and train_task in ["SST-2", "mr", "cr"]:
        templates = [t.replace(".", " .") for t in templates]

    max_length_per_example = max_length

    if use_demonstrations and not ensemble:
        assert do_zeroshot and not do_train
        mem = batch_size * max_length
        if n_classes == 2:
            max_length = max_length * k
        elif n_classes in [4, 5]:
            max_length = int(max_length * 1.5 * k)
        elif n_classes in [6]:
            max_length = int(max_length * 2 * k)
        else:
            max_length = 1024

        max_length = min(max_length, 1024)
        batch_size = int(mem / max_length)

    if do_zeroshot:
        cache_paths = [get_paths(out_dir, gpt2, method_type, task, do_zeroshot,
                                 k, seed, train_seed, split, template_idx,
                                 use_demonstrations=use_demonstrations,
                                 ensemble=ensemble)]
        checkpoints = [None]

    else:
        out_dir = get_paths(out_dir, gpt2, method_type, train_task, do_zeroshot,
                            k, seed, train_seed, split, template_idx,
                            batch_size, learning_rate, warmup_steps,
                            use_demonstrations=use_demonstrations,
                            ensemble=ensemble,
                            prompt_tune=prompt_tune,
                            head_tune=head_tune,
                            transform_tune=transform_tune,
                            prior_tune=prior_tune,
                            n_prefix=n_prefix)

        k = int(k)
        eval_period = 100
        num_training_steps = 400 if k != 16384 else 1000

        cache_paths = [os.path.join(out_dir, "{}cache-{}-{}.pkl".format(
            task + "-" if train_task != task else "",
            split, step))
                       for step in range(num_training_steps, 0, -eval_period)]
        checkpoints = [os.path.join(out_dir, "model-{}.pt".format(step))
                       for step in range(num_training_steps, 0, -eval_period)]

    mapping = None

    if do_train and (head_tune or not do_check):

        inputs = prepare_data(
            tokenizer, None, train_data,
            max_length=max_length,
            max_length_per_example=max_length_per_example,
            n_classes=n_classes_train,
            templates=templates_train,
            method_type=method_type,
            is_training=True,
            ensemble=ensemble)

        logger.info(out_dir)

        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)
            except:
                pass

        if not do_check:

            if deep_speed:
                dschf = HfDeepSpeedConfig(ds_config)

            model = GPT2LMHeadModel.from_pretrained(gpt2)

            if prior_tune:
                for param in model.parameters():
                    param.requires_grad = False
                if prior_weight < 0:
                    set_prior(model, n_classes, 1.0)
                    model.lm_head.gamma.requires_grad = True
                else:
                    set_prior(model, n_classes, prior_weight)
                    model.lm_head.gamma.requires_grad = False
                model.lm_head.priors.requires_grad = True 
                if prompt_tune:
                    set_extra_embeddings(model, n_prefix)
                    inputs = prepend_task_tokens(tokenizer, inputs, n_prefix)

            elif prompt_tune:
                for param in model.parameters():
                    param.requires_grad = False

                set_extra_embeddings(model, n_prefix)
                inputs = prepend_task_tokens(tokenizer, inputs, n_prefix)

            elif head_tune:
                mapping, inputs = reassign_output_tokens(inputs, for_labels=True)
                logger.info("Created mapping with {} vocabs".format(len(mapping)))
                set_separate_lm_head(model, mapping)
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.lm_head.my_lm_head.parameters():
                    param.requires_grad = True

            elif transform_tune:
                set_transformed_lm_head(model)
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.lm_head.transform.parameters():
                    param.requires_grad = True

            model = model.cuda()

            # if torch.cuda.device_count() > 1 and not deep_speed:
            #     model = torch.nn.DataParallel(model) 
            
            # distributed
            if local_rank >= 0 and not deep_speed:
                torch.distributed.init_process_group("nccl", init_method='env://')
                model = torch.nn.parallel.DistributedDataParallel(model,
                                                device_ids=[local_rank],
                                                output_device=local_rank)

            train(logger, model, inputs, batch_size, out_dir, ds_config, max(local_rank, 0),
                  learning_rate=learning_rate,
                  regularization_weight=regularization_weight,
                  warmup_steps=warmup_steps,
                  eval_period=eval_period,
                  num_training_steps=num_training_steps,
                  prompt_tune=prompt_tune,
                  head_tune=head_tune,
                  transform_tune=transform_tune,
                  prior_tune = prior_tune,
                  deep_speed=deep_speed)

    if local_rank > 0:
        logger.info("rank {} exited!".format(local_rank))
        exit()

    input_tensors = prepare_data(
        tokenizer, train_data, dev_data,
        max_length=max_length,
        max_length_per_example=max_length_per_example,
        n_classes=n_classes,
        templates=templates,
        method_type=method_type,
        use_demonstrations=use_demonstrations,
        ensemble=ensemble,
        is_null=is_null)

    if prompt_tune:
        input_tensors = prepend_task_tokens(tokenizer, input_tensors, n_prefix)

    if head_tune:
        # some tricks in case train_task and test_task are different
        # TODO: maybe need to consider this for the newly added datasets
        if task != train_task:
            if task in ["sst-5", "yelp_full", "amazon"] and train_task in ["SST-2", "mr", "cr"]:
                input_tensors = [input_tensors[0], input_tensors[-1]]
                if head_tune:
                    label_counter = {'0': '0', '4': '1'}
                    dev_data = [(x, label_counter.get(y, '-1')) for x, y in dev_data]
            elif task in ["SST-2", "mr"] and train_task in ["SST-2", "mr", "sst-5"]:
                pass
            else:
                raise NotImplementedError()

        if mapping is None:
            mapping, inputs = reassign_output_tokens(inputs, for_labels=head_tune)

        train_labels = set([label for _, label in train_data])
        if len(train_labels) != n_classes:
            train_labels = sorted(train_labels)
            input_tensors = [input_tensors[int(l)] for l in train_labels]
            dev_data = [(sent, str(train_labels.index(l)) if l in train_labels else -1)
                        for sent, l in dev_data]

        _, input_tensors = reassign_output_tokens(input_tensors, for_labels=head_tune,
                                                  mapping={v: k for k, v in mapping.items()})
        logger.info(mapping)
        logger.info("Checked that train mapping and test mapping are identical")


    # for debugging ...
    logger.info("Checking the first example...")
    input_ids = input_tensors[0]["input_ids"][0].numpy().tolist()
    token_type_ids = input_tensors[0]["token_type_ids"][0].numpy().tolist()
    logger.info("Input:")
    logger.info(tokenizer.decode(input_ids[:token_type_ids.index(1)]))
    logger.info("Output:")
    logger.info(tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1]))

    results = []
    for cache_path, checkpoint in zip(cache_paths, checkpoints):

        logger.info(cache_path)

        # if there is a cache, load it
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                losses = pkl.load(f)
        else:
            if checkpoint is not None and not os.path.exists(checkpoint):
                logger.info("checkpoint %s not found..." % checkpoint)
                assert False

            if checkpoint is None and model is not None and do_zeroshot:
                logger.info("Reusing the loaded model...")
                pass
            else:
                logger.info("Loading the model")
                torch.cuda.empty_cache()
                del model
                model = load_checkpoint(gpt2, checkpoint,
                                        prompt_tune=prompt_tune,
                                        head_tune=head_tune,
                                        transform_tune=transform_tune,
                                        prior_tune=prior_tune,
                                        n_prefix=n_prefix,
                                        mapping=mapping,
                                        n_classes=n_classes)

                # For debugging
                if prior_tune:
                    logger.info("The prior's value are: {}".format(model.lm_head.priors.tolist()))
                    logger.info("The weight for priors is: {}".format(model.lm_head.gamma.item()))

                model = model.cuda()
                model.eval()
                logger.info("Finished loading the model")

            losses = []
            for input_tensor in input_tensors:
                losses.append(inference(model,
                                        input_tensor,
                                        batch_size))

            with open(cache_path, "wb") as f:
                pkl.dump(losses, f)

        if is_null:
            continue

        if ensemble:
            losses = flatten_label_losses(losses, dev_data)

        if use_calibration:
            bias_path = cache_path.replace(split, "None")
            assert os.path.exists(bias_path), bias_path
            with open(bias_path, "rb") as f:
                bias_losses = pkl.load(f)

            for i, (bias_loss, loss) in enumerate(zip(bias_losses, losses)):
                loss = np.array(loss)
                bias_loss = np.array(bias_loss)
                if ensemble:
                    bias_loss = bias_loss.reshape(1, -1)
                losses[i] = loss - bias_loss


        acc, f1 = evaluate(dev_data, {str(i): loss for i, loss in enumerate(losses)})
        logger.info(acc)
        logger.info(f1)
        return acc, f1
    return None, None

def evaluate(dev_data, label_losses, is_classification=True):
    if type(label_losses)==list:
        label_losses = np.array(label_losses)
    accs = []
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    for idx, (_, label) in enumerate(dev_data):
        label_loss = {l:np.sum(label_losses[l][idx]) for l in label_losses}
        prediction = sorted(label_loss.items(), key=lambda x: x[1])[0][0]
        accs.append(prediction==label)
        precisions[prediction].append(prediction==label)
        recalls[label].append(prediction==label)

    if not is_classification:
        return np.mean(accs)

    f1s = []
    for key in recalls:
        precision = np.mean(precisions[key]) if key in precisions else 1.0
        recall = np.mean(recalls[key])
        if precision+recall==0:
            f1s.append(0)
        else:
            f1s.append(2*precision*recall / (precision+recall))

    return np.mean(accs), np.mean(f1s)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_zeroshot", default=False, action="store_true")
    parser.add_argument("--do_check", default=False, action="store_true")

    parser.add_argument("--use_calibration", default=False, action="store_true")
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--ensemble", default=False, action="store_true")
    parser.add_argument("--prompt_tune", default=False, action="store_true")
    parser.add_argument("--head_tune", default=False, action="store_true")
    parser.add_argument("--transform_tune", default=False, action="store_true")
    parser.add_argument("--prior_tune", default=False, action="store_true")

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default="SST-2")
    parser.add_argument("--train_task", type=str, default=None)

    parser.add_argument("--k", type=str, default="16")
    parser.add_argument("--seed", type=str, default="100")
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--prior_weight", type=float, default=1.0)
    parser.add_argument("--regularization_weight", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="out")

    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--method", type=str, default="direct")
    parser.add_argument("--n_prefix", type=int, default=20)
    parser.add_argument("--gpt2", type=str, default="gpt2-large")

    parser.add_argument("--deep_speed", default=False, action="store_true")
    parser.add_argument("--local_rank", type=str)

    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
