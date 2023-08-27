import datetime
import sys

from openprompt.utils.reproduciblity import set_seed

from tqdm import tqdm
import os
import torch
from openprompt.prompts import SoftVerbalizer, ManualTemplate

from models import HierICRFPromptForHTC
from processor import PROCESSOR

from util.utils import parse_args, load_plm_from_config
from util.data_loader import CHRPromptDataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from util.utils import print_info, get_template

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

use_cuda = True

MODEL_DICT = dict({
    "hierICRF": HierICRFPromptForHTC
})


def main():
    args = parse_args()
    start_time = datetime.datetime.now()

    args.result_file = f"{args.model}.txt"
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if args.device != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
        device = torch.device("cuda:0")
        use_cuda = True
    else:
        use_cuda = False
        device = torch.device("cpu")
    if not os.path.exists("ckpts"):
        os.mkdir("ckpts")

    processor = PROCESSOR[args.dataset](shot=args.shot, seed=args.seed)

    train_data = processor.train_example
    dev_data = processor.dev_example
    test_data = processor.test_example
    train_data = [[i.text_a, i.label] for i in train_data]
    dev_data = [[i.text_a, i.label] for i in dev_data]
    test_data = [[i.text_a, i.label] for i in test_data]
    hier_mapping = processor.hier_mapping
    args.depth = len(hier_mapping) + 1

    print_info("final train_data length is: {}".format(len(train_data)))
    print_info("final dev_data length is: {}".format(len(dev_data)))
    print_info("final test_data length is: {}".format(len(test_data)))

    set_seed(args.seed)

    plm, tokenizer, model_config, WrapperClass = load_plm_from_config(args, args.model_name_or_path)
    ## dataset
    dataset = {}
    dataset['train'] = processor.train_example
    dataset['dev'] = processor.dev_example
    dataset['test'] = processor.test_example

    if args.multi_mask:
        template_file = f"{args.dataset}_mask_template_{args.template_id}.txt"
        if args.template_id == 6:
            template_file = f"{args.dataset}_mask_template_{args.template_id}_iter_num{args.iter_num}.txt"
    else:
        template_file = "manual_template.txt"
    template_path = "template"
    text = get_template(args)
    if not os.path.exists(template_path):
        os.mkdir(template_path)

    template_path = os.path.join(template_path, template_file)
    with open(template_path, 'w', encoding='utf-8') as fp:
        fp.write(text)
    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(template_path, choice=0)

    print_info("train_size: {}".format(len(dataset['train'])))
    if args.dataset == "wos":
        full_name = "WebOfScience"
    elif args.dataset == "dbp":
        full_name = "DBPedia"
    else:
        raise NotImplementedError
    ## Loading dataset
    if args.shot > 0:
        train_dataloader = CHRPromptDataLoader(dataset=dataset['train'], template=mytemplate,
                                               tokenizer=tokenizer,
                                               tokenizer_wrapper_class=WrapperClass,
                                               max_seq_length=args.max_seq_lens,
                                               decoder_max_length=3,
                                               batch_size=args.batch_size, shuffle=args.shuffle,
                                               teacher_forcing=False,
                                               predict_eos_token=False, truncate_method="tail",
                                               num_works=2,
                                               multi_gpu=(args.device == -2), )
    else:
        train_path = os.path.join(f"dataset", full_name, f"train_dataloader-multi_mask.pt")
        if args.template_id == 6:
            train_path = os.path.join("dataset", full_name,
                                      f"train_dataloader-multi_mask_template_id{args.template_id}_iter_num{args.iter_num}.pt")
        if os.path.exists(train_path):
            train_dataloader = torch.load(train_path)
        else:
            train_dataloader = CHRPromptDataLoader(dataset=dataset['train'], template=mytemplate,
                                                   tokenizer=tokenizer,
                                                   tokenizer_wrapper_class=WrapperClass,
                                                   max_seq_length=args.max_seq_lens,
                                                   decoder_max_length=3,
                                                   batch_size=args.batch_size, shuffle=args.shuffle,
                                                   teacher_forcing=False,
                                                   predict_eos_token=False, truncate_method="tail",
                                                   num_works=2,
                                                   multi_gpu=(args.device == -2), )
            torch.save(train_dataloader, train_path)

    dev_path = os.path.join("dataset", full_name, f"dev_dataloader-multi_mask_template_id{args.template_id}.pt")
    test_path = os.path.join(f"dataset", full_name, f"test_dataloader-multi_mask_template_id{args.template_id}.pt")

    is_cover = False

    if args.template_id == 6:
        dev_path = os.path.join("dataset", full_name,
                                f"dev_dataloader-multi_mask_template_id{args.template_id}_iter_num{args.iter_num}.pt")
        test_path = os.path.join(f"dataset", full_name,
                                 f"test_dataloader-multi_mask_template_id{args.template_id}_iter_num{args.iter_num}.pt")
    if args.dataset != "dbp" and os.path.exists(dev_path) and not is_cover:
        validation_dataloader = torch.load(dev_path)
    else:
        validation_dataloader = CHRPromptDataLoader(dataset=dataset["dev"], template=mytemplate,
                                                    tokenizer=tokenizer,
                                                    tokenizer_wrapper_class=WrapperClass,
                                                    max_seq_length=args.max_seq_lens,
                                                    decoder_max_length=3,
                                                    batch_size=args.eval_batch_size, shuffle=False,
                                                    teacher_forcing=False,
                                                    predict_eos_token=False,
                                                    truncate_method="tail",
                                                    multi_gpu=False,
                                                    )
        if args.dataset != "dbp":
            torch.save(validation_dataloader, dev_path)
    if not os.path.exists(test_path) or is_cover:
        test_dataloader = CHRPromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                              tokenizer_wrapper_class=WrapperClass,
                                              max_seq_length=args.max_seq_lens,
                                              decoder_max_length=3,
                                              batch_size=args.eval_batch_size, shuffle=False,
                                              teacher_forcing=False,
                                              predict_eos_token=False,
                                              truncate_method="tail",
                                              multi_gpu=False,
                                              mode='test',
                                              )
        torch.save(test_dataloader, test_path)
    else:
        test_dataloader = torch.load(test_path)

    ## build verbalizer and model

    print_info("loading prompt model")

    if args.model in ['hierCRF', 'hierVerb']:
        verbalizer_list = []
        label_list = processor.label_list

        for i in range(args.depth):
            verbalizer_list.append(SoftVerbalizer(tokenizer, plm=plm, classes=label_list[i]))
        prompt_model = MODEL_DICT[args.model](plm=plm, template=mytemplate, verbalizer_list=verbalizer_list,
                                              freeze_plm=args.freeze_plm, args=args, processor=processor,
                                              plm_eval_mode=args.plm_eval_mode, use_cuda=use_cuda)
    else:
        myverbalizer = SoftVerbalizer(tokenizer, model=plm, classes=processor.all_labels)
        prompt_model = MODEL_DICT[args.model](plm=plm, template=mytemplate, verbalizer=myverbalizer,
                                              freeze_plm=args.freeze_plm,
                                              plm_eval_mode=args.plm_eval_mode, args=args, processor=processor)
    if use_cuda:
        prompt_model = prompt_model.cuda()
    print("prompt_model.device", prompt_model.device)
    ## Prepare training parameters
    # it's always good practice to set no decay to biase and LayerNorm parameters
    no_decay = ['bias', 'LayerNorm.weight']

    named_parameters = prompt_model.plm.named_parameters()

    optimizer_grouped_parameters1 = [

        {'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters
    if args.model == "hierICRF":
        # softVerb
        optimizer_grouped_parameters2 = [
            {'params': prompt_model.verbalizer.group_parameters_1, "lr": args.lr},
            {'params': prompt_model.verbalizer.group_parameters_2, "lr": args.lr2},
        ]
        # if args.shot > 0:
        #     args.max_epochs = 20
    else:
        raise NotImplementedError
    if hasattr(prompt_model, "HierCRF"):
        optimizer_grouped_parameters2.append({'params': prompt_model.HierCRF_parameters_3(), "lr": args.lr3}, )

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.lr, eps=args.adam_epsilon)

    optimizer2 = AdamW(optimizer_grouped_parameters2, eps=args.adam_epsilon)

    # optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.lr)
    #
    # optimizer2 = AdamW(optimizer_grouped_parameters2)

    # optimizer3 = AdamW(optimizer_grouped_parameters3)
    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs

    warmup_steps = 0
    scheduler1 = None
    scheduler2 = None
    if args.use_scheduler1:
        scheduler1 = get_linear_schedule_with_warmup(
            optimizer1,
            num_warmup_steps=warmup_steps, num_training_steps=tot_step)
    if args.use_scheduler2:
        scheduler2 = get_linear_schedule_with_warmup(
            optimizer2,
            num_warmup_steps=warmup_steps, num_training_steps=tot_step)
    print_info(args)
    best_score_macro = 0
    best_score_micro = 0

    corr_best_score_macro = 0
    corr_best_score_micro = 0

    best_score_macro_epoch = -1
    best_score_micro_epoch = -1
    early_stop_count = 0

    this_run_unicode = f"{args.dataset}-seed{args.seed}-shot{args.shot}-template_id{args.template_id}_iter_num{args.iter_num}freeze_lm{args.freeze_plm}-lr1{args.lr}-lr2{args.lr2}-batch_size{args.batch_size}-shuffle{args.shuffle}-lm_loss{args.lm_training}-multi_verb{args.multi_verb}"
    print_info("saved_path: {}".format(this_run_unicode))

    if args.eval_full:
        best_record = dict()
        keys = ['p_micro_f1', 'p_macro_f1', 'c_micro_f1', 'c_macro_f1', 'P_acc']
        for key in keys:
            best_record[key] = 0

    # torch.autograd.set_detect_anomaly(True)
    ## start training
    if args.do_train:
        prompt_model.zero_grad()
        for epoch in range(args.max_epochs):
            # for epoch in range(11):
            print_info("------------ epoch {} ------------".format(epoch + 1))
            if early_stop_count >= args.early_stop != -1:
                print_info("Early stop!")
                break

            print_info(
                f"cur lr"
                f"\tscheduler1: {scheduler1.get_lr() if scheduler1 is not None else args.lr}"
                f"\tscheduler2: {scheduler2.get_lr() if scheduler2 is not None else args.lr2}"
            )

            loss_detailed = [0, 0, 0, 0]
            prompt_model.train()
            idx = 0

            for batch in tqdm(train_dataloader):
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                batch = {"input_ids": batch[0], "attention_mask": batch[1],
                         "label": batch[2], "loss_ids": batch[3]}

                logits, loss, cur_loss_detailed = prompt_model(batch)
                loss_detailed = [loss_detailed[idx] + value for idx, value in enumerate(cur_loss_detailed)]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), args.max_grad_norm)

                optimizer1.step()
                optimizer2.step()

                if scheduler1 is not None:
                    scheduler1.step()
                if scheduler2 is not None:
                    scheduler2.step()

                prompt_model.zero_grad()

                idx = idx + 1

            if args.model == "hierCRF":
                print_info("multi-verb loss, lm loss, hierCRF loss, none are: ")
            elif args.model == "hierVerb":
                print_info("multi-verb loss, lm loss, constraint loss, contrastive loss are: ")
            elif args.model == "softVerb":
                print_info("single-verb loss, lm loss, hierCRF loss, none are: ")
            print_info(loss_detailed)
            if args.evaluate_both:
                scores, scores2 = prompt_model.evaluate(validation_dataloader, processor, desc="Valid",
                                                        mode=args.eval_mode, device=device, args=args)
            else:
                scores = prompt_model.evaluate(validation_dataloader, processor, desc="Valid",
                                               mode=args.eval_mode, device=device)
            early_stop_count += 1
            if args.eval_full:
                score_str = ""
                for key in keys:
                    score_str += f'{key} {scores[key]}\n'
                print_info(score_str)
                for k in best_record:
                    if scores[k] > best_record[k]:
                        best_record[k] = scores[k]
                        torch.save(prompt_model.state_dict(), f"ckpts/{this_run_unicode}-{k}.ckpt")
                        early_stop_count = 0

            else:
                macro_f1 = scores['macro_f1']
                micro_f1 = scores['micro_f1']

                print_info('macro {} micro {}'.format(macro_f1, micro_f1))
                if args.evaluate_both:
                    macro2_f1 = scores2['macro_f1']
                    micro2_f1 = scores2['micro_f1']
                    print_info('macro2 {} micro2 {}'.format(macro2_f1, micro2_f1))
                if macro_f1 > best_score_macro:
                    best_score_macro = macro_f1
                    corr_best_score_micro = micro_f1
                    torch.save(prompt_model.state_dict(), f"ckpts/{this_run_unicode}-macro.ckpt")
                    # save(macro_f1, best_score_macro, os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
                    early_stop_count = 0
                    best_score_macro_epoch = epoch + 1

                if micro_f1 > best_score_micro:
                    best_score_micro = micro_f1
                    corr_best_score_macro = macro_f1
                    torch.save(prompt_model.state_dict(), f"ckpts/{this_run_unicode}-micro.ckpt")
                    # save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
                    early_stop_count = 0
                    best_score_micro_epoch = epoch + 1

    print_info(
        'finally best macro at epoch({}) macro {} micro {}'.format(best_score_macro_epoch, best_score_macro,
                                                                   corr_best_score_micro))
    print_info(
        'finally best micro at epoch({}) macro {} micro {}'.format(best_score_micro_epoch, corr_best_score_macro,
                                                                   best_score_micro))

    ## evaluate
    if args.do_test:
        print("begin evaluate")
        if args.eval_full:
            best_keys = ['P_acc']
            for k in best_keys:
                if os.path.exists(f"ckpts/{this_run_unicode}-{k}.ckpt"):
                    prompt_model.load_state_dict(torch.load(f"ckpts/{this_run_unicode}-{k}.ckpt"))

                scores = prompt_model.evaluate(test_dataloader, processor, desc="test", mode=args.eval_mode,
                                               args=args, device=device)
                tmp_str = ''
                tmp_str += f"finally best_{k} "
                for i in keys:
                    tmp_str += f"{i}: {scores[i]}\t"
                print_info(tmp_str)

        else:
            # for best macro
            if args.do_train:
                prompt_model.load_state_dict(torch.load(f"ckpts/{this_run_unicode}-macro.ckpt"))

                if use_cuda:
                    prompt_model = prompt_model.cuda()
                if not args.evaluate_both:
                    scores = prompt_model.evaluate(test_dataloader, processor, desc="Test", mode=args.eval_mode,
                                                   device=device)
                else:
                    scores, scores2 = prompt_model.evaluate(test_dataloader, processor, desc="Test",
                                                            mode=args.eval_mode,
                                                            device=device, args=args)

                macro_f1_1 = scores['macro_f1']
                micro_f1_1 = scores['micro_f1']

                print_info(
                    'macro  {} micro {}'.format(macro_f1_1, micro_f1_1))
                if args.evaluate_both:
                    macro2_f1 = scores2['macro_f1']
                    micro2_f1 = scores2['micro_f1']
                    print_info(
                        'macro2  {} micro2 {}'.format(macro2_f1, micro2_f1))

                # for best micro

                prompt_model.load_state_dict(torch.load(f"ckpts/{this_run_unicode}-micro.ckpt"))
                if not args.evaluate_both:
                    scores = prompt_model.evaluate(test_dataloader, processor, desc="Test", mode=args.eval_mode,
                                                   device=device)
                else:
                    scores, scores2 = prompt_model.evaluate(test_dataloader, processor, desc="Test",
                                                            mode=args.eval_mode,
                                                            device=device, args=args)
                macro_f1_2 = scores['macro_f1']
                micro_f1_2 = scores['micro_f1']

                print_info(
                    'macro {} micro {}'.format(macro_f1_2, micro_f1_2))
                if args.evaluate_both:
                    macro2_f1 = scores2['macro_f1']
                    micro2_f1 = scores2['micro_f1']
                    print_info(
                        'macro2  {} micro2 {}'.format(macro2_f1, micro2_f1))
            else:
                if use_cuda:
                    prompt_model = prompt_model.cuda()
                print_info(f"prompt_model.device: {prompt_model.device}", )
                scores = prompt_model.evaluate(test_dataloader, processor, desc="Test", mode=args.eval_mode,
                                               device=device)
                macro_f1 = scores['macro_f1']
                micro_f1 = scores['micro_f1']
                print_info('zero-shot inference micro {} micro {}'.format(best_score_micro_epoch, macro_f1, micro_f1))

        ## print and record parameter details
        content_write = "=" * 20 + "\n"
        content_write += f"start_time {start_time}" + "\n"
        content_write += f"end_time {datetime.datetime.now()}\t"
        for hyperparam, value in args.__dict__.items():
            content_write += f"{hyperparam} {value}\t"
        content_write += "\n"
        if args.eval_full:
            cur_keys = ['P_acc']
            for key in cur_keys:
                content_write += f"best_{key} "
                for i in keys:
                    content_write += f"{i}: {best_record[i]}\t"
                content_write += f"\n"
        else:
            if args.do_train:
                content_write += f"best_macro macro_f1: {macro_f1_1}\t"
                content_write += f"micro_f1: {micro_f1_1}\t"
                content_write += "\n"
                content_write += f"best_micro macro_f1: {macro_f1_2}\t"
                content_write += f"micro_f1: {micro_f1_2}\t"
            else:
                content_write += f"macro_f1: {macro_f1}\t"
                content_write += f"micro_f1: {micro_f1}\t"

        content_write += "\n\n"

        print_info(content_write)
        if not os.path.exists("result"):
            os.mkdir("result")
        with open(os.path.join("result", args.result_file), "a") as fout:
            fout.write(content_write)

    os.remove(f"ckpts/{this_run_unicode}-macro.ckpt")
    os.remove(f"ckpts/{this_run_unicode}-micro.ckpt")


if __name__ == "__main__":
    main()
