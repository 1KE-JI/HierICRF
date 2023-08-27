# -*- coding:utf-8 -*-
import torch
import numpy as np
from transformers import __version__ as transformers_version
import random
from transformers import BertTokenizer, BertPreTrainedModel, BertForSequenceClassification

from transformers import BertConfig
from openprompt.plms.mlm import MLMTokenizerWrapper
import argparse

logger = None


def print_info(info, file=None):
    if logger is not None:
        logger.info(info)
    else:
        print(info, file=file)


def parse_args(model="hierICRF"):
    parser = argparse.ArgumentParser("")

    parser.add_argument("--model", type=str, default=model, choices=['hierICRF', 'hierVerb'])
    parser.add_argument("--model_name_or_path", default='bert-base-uncased')
    parser.add_argument("--evaluate_both", default=0, type=int)
    parser.add_argument("--test_during_train", default=0, type=int)
    parser.add_argument("--template_id", default=6, type=int)
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--seed", type=int, default=550)
    parser.add_argument("--freeze_plm", default=0, type=int)
    parser.add_argument("--use_hier_mean", default=0, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--lr2", default=5e-5, type=float)
    parser.add_argument("--lr3", default=5e-5, type=float)

    parser.add_argument("--eval_mode", default=1, type=int)
    # lora config
    parser.add_argument("--apply_lora", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=8)

    parser.add_argument("--result_file", type=str, default="HierICRF.txt")
    parser.add_argument("--multi_mask", type=int, default=1)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--shuffle", default=1, type=int)

    parser.add_argument('--loss_type', default='focal', type=str,
                        choices=['lsr', 'focal', 'ce'])
    parser.add_argument('--focal_gamma', default=2, type=int)
    parser.add_argument("--do_train", default=1, type=int)
    parser.add_argument("--do_dev", default=1, type=bool)
    parser.add_argument("--do_test", default=1, type=bool)

    parser.add_argument("--not_manual", default=False, type=int)
    parser.add_argument("--depth", default=2, type=int)

    parser.add_argument("--device", default=0, type=int)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--dataset", default="wos", type=str)

    parser.add_argument("--multi_verb", default=0, type=int)

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_seq_lens", default=512, type=int, help="Max sequence length.")
    parser.add_argument("--use_withoutWrappedLM", default=False, type=bool)
    parser.add_argument('--mean_verbalizer', default=True, type=bool)

    parser.add_argument("--plm_eval_mode", default=False)
    parser.add_argument("--verbalizer", type=str, default="soft")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--multi_label", default=0, type=int)

    parser.add_argument("--early_stop", default=10, type=int)
    parser.add_argument("--eval_full", default=0, type=int)

    parser.add_argument("--apply_transitions", default=0, type=int)
    parser.add_argument("--apply_transitions_only_impossible", default=1, type=int)
    if model == "hierVerb":

        parser.add_argument("--use_new_ct", default=1, type=int)
        parser.add_argument("--contrastive_loss", default=1, type=int)
        parser.add_argument("--contrastive_level", default=1, type=int)
        parser.add_argument("--contrastive_alpha", default=0.99, type=float)
        parser.add_argument("--contrastive_logits", default=1, type=int)
        parser.add_argument("--use_dropout_sim", default=1, type=int)
        parser.add_argument("--imbalanced_weight", default=True, type=bool)
        parser.add_argument("--imbalanced_weight_reverse", default=True, type=bool)
        parser.add_argument("--max_epochs", type=int, default=20)
        parser.add_argument("--constraint_loss", default=1, type=int)
        parser.add_argument("--constraint_alpha", default=-1, type=float)
        parser.add_argument("--cs_mode", default=0, type=int)

        parser.add_argument("--lm_training", default=1, type=int)
        parser.add_argument("--lm_alpha", default=0.999, type=float)

        # parser.add_argument("--lr", default=5e-5, type=float)
        # parser.add_argument("--lr2", default=1e-4, type=float)

        parser.add_argument("--use_scheduler1", default=1, type=int)
        parser.add_argument("--use_scheduler2", default=1, type=int)

        parser.add_argument("--batch_size", default=5, type=int)
        parser.add_argument("--eval_batch_size", default=20, type=int)

    elif model == "hierCRF":

        # parser.add_argument("--lr", default=5e-5, type=float)
        # parser.add_argument("--lr2", default=1e-4, type=float)
        # parser.add_argument("--lr3", default=5e-2, type=float)
        # parser.add_argument("--lr2", default=1e-4, type=float)
        # parser.add_argument("--lr3", default=1e-2, type=float)

        parser.add_argument("--use_scheduler1", default=1, type=int)
        parser.add_argument("--use_scheduler2", default=1, type=int)

        parser.add_argument("--max_epochs", type=int, default=50)
        parser.add_argument("--hierCRF_loss", default=0, type=int)

        parser.add_argument("--hierCRF_alpha", default=-1, type=float)
        parser.add_argument("--batch_size", default=5, type=int)
        parser.add_argument("--eval_batch_size", default=5, type=int)

        parser.add_argument("--multi_verb_loss", default=0, type=int)
        parser.add_argument("--multi_verb_loss_alpha", default=-1, type=int)

        parser.add_argument("--lm_training", default=0, type=int)
        parser.add_argument("--lm_alpha", default=0.999, type=float)

    elif model == "hierICRF":

        parser.add_argument("--use_scheduler1", default=1, type=int)
        parser.add_argument("--use_scheduler2", default=1, type=int)

        parser.add_argument("--iter_num", default=5, type=int)

        parser.add_argument("--max_epochs", type=int, default=20)

        parser.add_argument("--batch_size", default=8, type=int)
        parser.add_argument("--eval_batch_size", default=16, type=int)

        parser.add_argument("--lm_training", default=0, type=int)
        parser.add_argument("--lm_alpha", default=0.999, type=float)

        parser.add_argument("--hierCRF_loss", default=1, type=int)

        parser.add_argument("--hierCRF_alpha", default=-1, type=float)
    else:
        raise NotImplementedError
    args = parser.parse_args()
    return args


def get_template(args):
    if args.template_id == 0:
        text_mask = []
        for i in range(args.depth):
            text_mask.append(f'{i + 1} level: {{"mask"}}')
        text = f'It was {" ".join(text_mask)}. {{"placeholder": "text_a"}}'
    elif args.template_id == 1:
        text_mask = []
        for i in range(args.depth):
            text_mask.append(f'{{"mask"}}')
        text = f'It was {" ".join(text_mask)}. {{"placeholder": "text_a"}}'
    elif args.template_id == 2:
        text_mask = []
        for i in range(args.depth):
            text_mask.append(f'{i + 1} layer: {{"mask"}}')
        text = f'It was {" ".join(text_mask)}. {{"placeholder": "text_a"}}'
    elif args.template_id == 3:
        text_mask = []
        for i in range(args.depth):
            text_mask.append(f'{i + 1} level: {{"mask"}}')
        text = f'{{"placeholder": "text_a"}} This topic is about {" ".join(text_mask)}. '
    elif args.template_id == 4:
        text_mask = []
        for i in range(args.depth):
            text_mask.append(f'{{"mask"}}')
        text = f'{{"placeholder": "text_a"}}. Its topics from coarse-grained to fine-grained are {" ".join(text_mask)}. '
    elif args.template_id == 5:
        text_mask = []
        for i in range(args.depth):
            text_mask.append(f'{i + 1} level: {{"mask"}}')
        text = f'{{"placeholder": "text_a"}}. Its topics from coarse-grained to fine-grained are {" ".join(text_mask)}. '
    elif args.template_id == 6:
        text_mask = []
        for i in range(args.depth):
            text_mask.append(f'{i + 1} level: {{"mask"}}')
        for i in range(args.depth - 2, -1, -1):
            text_mask.append(f'{i + 1} level: {{"mask"}}')
        for _ in range(1, args.iter_num):
            for i in range(args.depth-1, -1, -1):
                text_mask.append(f'{i + 1} level: {{"mask"}}')
        text = f'It was {" ".join(text_mask)}. {{"placeholder": "text_a"}}'
    else:
        raise NotImplementedError
    print("template:", text)
    return text


def load_plm_from_config(args, model_path, specials_to_add=None, **kwargs):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.

    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`wrapper`: The wrapper class of this plm.
    """
    trainable_params = []
    model_config = BertConfig.from_pretrained(model_path)
    if args.apply_lora:
        from lora_bert.modeling_bert import BertForMaskedLM
        args.weight_decay = 0.1
        trainable_params.append('lora')

        model_config.apply_lora = 1
        model_config.lora_alpha = 16
        model_config.lora_r = 8
    else:
        from transformers import BertForMaskedLM
    args.trainable_params = trainable_params

    # model_config.hidden_dropout_prob = args.dropout
    model = BertForMaskedLM.from_pretrained(model_path, config=model_config)

    tokenizer = BertTokenizer.from_pretrained(model_path)
    wrapper = MLMTokenizerWrapper

    return model, tokenizer, model_config, wrapper


def seed_torch(seed=1029):
    print('Set seed to', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _mask_tokens(tokenizer, input_ids):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = input_ids.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
    if [int(v) for v in transformers_version.split('.')][:3] >= [2, 4, 0]:
        ignore_value = -100
    else:
        ignore_value = -1

    labels[~masked_indices] = ignore_value  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return input_ids, labels
