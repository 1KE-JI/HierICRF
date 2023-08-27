import openprompt
import torch
from openprompt import PromptForClassification
from typing import List
from transformers.utils.dummy_pt_objects import PreTrainedModel
from tqdm import tqdm
from transformers import BertTokenizer
from util.utils import _mask_tokens, print_info
from util.eval import compute_score

from openprompt.prompt_base import Template, Verbalizer

from models.crf import HierCRF
from models.focal_loss import FocalLoss as FocalLoss

import torch.nn.functional as F

from transformers import BartForConditionalGeneration, BartModel, BartConfig, T5Model, T5Config
import torch.nn as nn
from typing import List, Optional, Tuple, Union


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class HierICRFPromptForHTC(PromptForClassification):
    def __init__(self, plm: PreTrainedModel, template: Template, verbalizer: Verbalizer, freeze_plm: bool = False,
                 plm_eval_mode: bool = False, args=None, processor=None):
        super().__init__(plm, template, verbalizer, freeze_plm, plm_eval_mode)
        self.processor = processor
        self.args = args
        if self.args.mean_verbalizer:
            self.init_embeddings()
        self.num_tags = len(self.processor.all_labels)
        self.flag_lm_loss = False
        self.flag_hierCRF_loss = False

        if args.multi_label:
            self.loss_func = torch.nn.BCEWithLogitsLoss()
        else:
            if self.args.loss_type == 'ce':
                self.loss_func = torch.nn.CrossEntropyLoss()
            elif self.args.loss_type == 'focal':
                self.loss_func = FocalLoss()
            else:
                raise NotImplementedError
        if self.args.hierCRF_loss:
            start_transitions, end_transitions, transitions = None, None, None
            if args.apply_transitions:
                start_transitions, end_transitions, transitions = self.init_transitions()
            self.HierCRF = HierCRF(len(self.processor.all_labels), batch_first=True,
                                   start_transitions=start_transitions,
                                   end_transitions=end_transitions, transitions=transitions)
        if self.args.apply_lora:
            if len(self.args.trainable_params) > 0:
                for name, param in self.plm.named_parameters():
                    if "bert" in name or "roberta" in name:
                        param.requires_grad = False
                        for trainable_param in self.args.trainable_params:
                            if trainable_param in name:
                                param.requires_grad = True
                                break
                    else:
                        param.requires_grad = True

    def init_transitions(self, init_range=1):
        from sentence_transformers import SentenceTransformer
        print("init_transitions")
        model = SentenceTransformer("bert-base-uncased")
        label_embs = []
        for depth_idx in range(self.args.depth):
            label_emb = model.encode(self.processor.label_list[depth_idx])
            label_emb = torch.from_numpy(label_emb)
            label_embs.append(label_emb)

        sim_list = []
        for depth_idx in range(self.args.depth - 1):
            ## cos_sim: num_tags(depth_idx) * num_tags(depth_idx+1)
            sim = F.cosine_similarity(label_embs[depth_idx].unsqueeze(1), label_embs[depth_idx + 1].unsqueeze(0),
                                      dim=-1)
            sim_list.append(sim)

        start_transitions = torch.empty(self.num_tags)
        end_transitions = torch.empty(self.num_tags)
        transitions = torch.empty(self.num_tags, self.num_tags)

        torch.nn.init.uniform_(start_transitions, -init_range, init_range)
        torch.nn.init.uniform_(end_transitions, -init_range, init_range)
        torch.nn.init.uniform_(transitions, -init_range, init_range)
        start_transitions[len(self.processor.label_list[0]):] = -init_range

        ## eye impossible transitions
        transitions[torch.eye(self.num_tags).bool()] = -init_range

        start_end_list = []
        start = 0
        for depth_idx in range(self.args.depth):
            if depth_idx == 0:
                start += 0
            else:
                start += len(self.processor.label_list[depth_idx - 1])
            end = start + len(self.processor.label_list[depth_idx])
            start_end_list.append([start, end])
        print("start_end_list", start_end_list)
        ## prior knowledge injection
        if not self.args.apply_transitions_only_impossible:
            for depth_idx in range(1, self.args.depth - 1):
                pre_start, pre_end = start_end_list[depth_idx - 1]
                cur_start, cur_end = start_end_list[depth_idx]
                transitions[pre_start: pre_end, cur_start: cur_end] = sim_list[depth_idx - 1]
        ## forward impossible probability
        for depth_idx in range(0, self.args.depth):
            if not (0 <= (depth_idx - 1) and (depth_idx + 1) <= (self.args.depth - 1)):
                continue
            pre_start, pre_end = start_end_list[depth_idx - 1]
            next_start, next_end = start_end_list[depth_idx + 1]
            transitions[pre_start: pre_end, next_start: next_end] = -init_range

        ## backward impossible probability
        for depth_idx in range(self.args.depth - 1, 0, -1):
            cur_start, cur_end = start_end_list[depth_idx]
            transitions[cur_start: cur_end, 0:cur_start] = -init_range

        ## same layer impossible probability
        for depth_id in range(self.args.depth - 1):
            if depth_id == 0:
                start += 0
            else:
                start += len(self.processor.label_list[depth_id - 1])
            end = start + len(self.processor.label_list[depth_id])
            transitions[start: end, start: end] = -init_range
        return start_transitions, end_transitions, transitions

    def forward(self, batch, type="none") -> torch.Tensor:
        r"""
        Get the logits of label words.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The logits of the lable words (obtained by the current verbalizer).
        """
        loss = 0
        loss_detailed = [0, 0, 0, 0]
        lm_loss = None
        hierCRF_loss = None
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        if isinstance(outputs, tuple):
            outputs_at_mask = [self.extract_at_mask(output, batch) for output in outputs]
        else:
            outputs_at_mask = self.extract_at_mask(outputs, batch)

        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        if type == "decode" and not self.training:
            return label_words_logits
        if self.training:

            labels = batch['label']

            hier_labels = []
            hier_labels.insert(0, labels)
            for idx in range(self.args.depth - 2, -1, -1):
                cur_depth_labels = torch.zeros_like(labels)
                for i in range(len(labels)):
                    # cur_depth_labels[i] = label1_to_label0_mapping[labels[i].tolist()]
                    cur_depth_labels[i] = self.processor.hier_mapping[idx][1][hier_labels[0][i].tolist()]
                hier_labels.insert(0, cur_depth_labels)

            new_hier_labels = []
            new_hier_labels.append(hier_labels[0])
            start = len(self.processor.depth2label[0])
            for depth_idx in range(1, self.args.depth):
                new_hier_labels.append(hier_labels[depth_idx] + start)
                start += len(self.processor.depth2label[depth_idx])
            hier_labels = new_hier_labels

            hier_labels = torch.stack(hier_labels).transpose(1, 0)

            if self.args.template_id == 6:
                iter_labels = torch.stack(
                    [torch.stack([hier_labels[i][j] for j in range(self.args.depth - 2, -1, -1)]) for i in
                     range(hier_labels.shape[0])])
                hier_labels = torch.cat([hier_labels, iter_labels], dim=hier_labels.dim() - 1)
                for _ in range(1, self.args.iter_num):
                    iter_labels = torch.stack(
                        [torch.stack([hier_labels[i][j] for j in range(self.args.depth - 1, -1, -1)]) for i in
                         range(hier_labels.shape[0])])
                    hier_labels = torch.cat([hier_labels, iter_labels], dim=hier_labels.dim() - 1)

            ## hierCRF loss
            if self.args.hierCRF_loss:
                if not self.flag_hierCRF_loss:
                    print_info("using hierCRF loss with alpha {}".format(self.args.hierCRF_alpha))
                    self.flag_hierCRF_loss = True
                hierCRF_loss = -1 * self.HierCRF(label_words_logits, hier_labels)
            else:
                loss += self.loss_func(label_words_logits, hier_labels)

                loss_detailed[0] += loss.item()  # 层级二loss
            ## MLM loss
            if self.args.lm_training:
                if not self.flag_lm_loss:
                    print_info("using lm loss with alpha {}".format(self.args.lm_alpha))
                    self.flag_lm_loss = True
                input_ids = batch['input_ids']
                input_ids, labels = _mask_tokens(self.tokenizer, input_ids.cpu())

                lm_inputs = {"input_ids": input_ids, "attention_mask": batch['attention_mask'], "labels": labels}

                for k, v in lm_inputs.items():
                    if v is not None:
                        lm_inputs[k] = v.to(self.device)
                lm_loss = self.plm(**lm_inputs)[0]

            if lm_loss is not None:
                if self.args.lm_alpha != -1:
                    loss = loss * self.args.lm_alpha + (1 - self.args.lm_alpha) * lm_loss
                else:
                    loss += lm_loss
                loss_detailed[1] += lm_loss.item()

            if hierCRF_loss is not None:

                if self.args.hierCRF_alpha != -1:
                    loss = loss * self.args.hierCRF_alpha + (1 - self.args.hierCRF_alpha) * hierCRF_loss
                else:
                    loss += hierCRF_loss
                loss_detailed[2] += hierCRF_loss.item()

            return label_words_logits, loss, loss_detailed
        else:
            return label_words_logits

    def evaluate(self, dataloader, processor, desc="Valid", mode=0, device="cuda:0", args=None, debug=False):
        self.eval()
        if self.args.hierCRF_loss:
            return self.evaluate_decode(dataloader, processor, desc=desc, mode=mode, device=device, args=args,
                                        debug=debug)
        preds = []
        truth = []
        pbar = tqdm(dataloader, desc=desc)
        hier_mapping = processor.hier_mapping
        depth = len(hier_mapping) + 1

        batch_s = 5
        for step, batch in enumerate(pbar):
            if hasattr(batch, 'cuda'):
                batch = batch.cuda()
            else:
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                batch = {"input_ids": batch[0], "attention_mask": batch[1],
                         "label": batch[2], "loss_ids": batch[3]}

            logits = self(batch)
            logits = torch.softmax(logits, dim=-1)
            cur_preds = torch.argmax(logits, dim=-1).cpu().tolist()

            leaf_labels = batch['label']
            hier_labels = []
            hier_labels.insert(0, leaf_labels)
            for idx in range(depth - 2, -1, -1):
                cur_depth_labels = torch.zeros_like(leaf_labels)
                for i in range(len(leaf_labels)):
                    # cur_depth_labels[i] = label1_to_label0_mapping[labels[i].tolist()]
                    cur_depth_labels[i] = hier_mapping[idx][1][hier_labels[0][i].tolist()]
                hier_labels.insert(0, cur_depth_labels)

            batch_golds = []
            for i in range(hier_labels[0].shape[0]):
                batch_golds.append([hier_labels[0][i].tolist(), (hier_labels[1][i] + 7).tolist()])
            for i in range(batch_s):
                preds.append(cur_preds[i])
                truth.append(batch_golds[i])

        label_dict = dict({idx: label for idx, label in enumerate(processor.all_labels)})

        scores = compute_score(preds, truth, label_dict)

        return scores

    def decode(self, batch):
        emissions = self(batch, type="decode")
        # batch_size * depth * num_tags(all_labels)

        return self.HierCRF.decode(emissions)

    def evaluate_decode(self, dataloader, processor, desc="Valid", mode=0, device="cuda:0", args=None, debug=False):
        self.eval()
        preds = []
        truth = []
        if args:
            preds2 = []
        pbar = tqdm(dataloader, desc=desc + "_hierCRF")
        hier_mapping = processor.hier_mapping
        depth = len(hier_mapping) + 1

        for step, batch in enumerate(pbar):
            if hasattr(batch, 'cuda'):
                batch = batch.cuda()
            else:
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                batch = {"input_ids": batch[0], "attention_mask": batch[1],
                         "label": batch[2], "loss_ids": batch[3]}

            flat_preds = self.decode(batch)
            if self.args.template_id == 6:
                if not args:
                    if mode == 0:
                        flat_preds = [pred[-self.args.depth:] for pred in flat_preds]
                        for i in flat_preds:
                            i.reverse()
                    elif mode == 1:
                        flat_preds = [pred[:self.args.depth] for pred in flat_preds]
                    else:
                        raise NotImplementedError
                else:
                    flat_preds1 = [pred[:self.args.depth] for pred in flat_preds]
                    flat_preds = [pred[-self.args.depth:] for pred in flat_preds]
                    for i in flat_preds:
                        i.reverse()

            leaf_labels = batch['label']
            batch_size = len(flat_preds)
            flat_preds = [list(set(flat_preds[i])) for i in range(batch_size)]
            if args:
                batch_size = len(flat_preds1)
                flat_preds1 = [list(set(flat_preds1[i])) for i in range(batch_size)]
            hier_labels = []
            hier_labels.insert(0, leaf_labels)
            for idx in range(depth - 2, -1, -1):
                cur_depth_labels = torch.zeros_like(leaf_labels)
                for i in range(len(leaf_labels)):
                    cur_depth_labels[i] = hier_mapping[idx][1][hier_labels[0][i].tolist()]
                hier_labels.insert(0, cur_depth_labels)

            batch_golds = []

            leaf_labels = leaf_labels.cpu().tolist()

            batch_golds.insert(0, leaf_labels)

            for depth_idx in range(depth - 2, -1, -1):
                cur_golds = hier_labels[depth_idx].cpu().tolist()

                batch_golds.insert(0, cur_golds)
            batch_golds = torch.tensor(batch_golds).transpose(1, 0).cpu().tolist()

            for i in range(batch_size):
                preds.append(flat_preds[i])
                if args:
                    preds2.append(flat_preds1[i])
            for i in range(batch_size):

                sub_golds = []
                prev_label_size = 0
                for depth_idx in range(depth):

                    if depth_idx == 0:
                        sub_golds.append(batch_golds[i][depth_idx])
                        continue
                    prev_mapping = hier_mapping[depth_idx - 1]
                    prev_label_size = len(prev_mapping[0]) + prev_label_size
                    sub_golds.append(batch_golds[i][depth_idx] + prev_label_size)
                truth.append(sub_golds)

        if debug:
            print(preds[:200])
            print(truth[:200])
        label_dict = dict({idx: label for idx, label in enumerate(processor.all_labels)})
        assert len(preds) == len(truth)
        scores = compute_score(preds, truth, label_dict, debug)
        if args:
            scores2 = compute_score(preds2, truth, label_dict, debug)
            return scores, scores2
        return scores

    def init_embeddings(self):

        if self.args.mean_verbalizer:
            print("using label emb for soft verbalizer")
            all_length = len(self.processor.all_labels)

            label_dict = dict({idx: v for idx, v in enumerate(self.processor.all_labels)})
            label_dict = {i: self.tokenizer.encode(v) for i, v in label_dict.items()}
            label_emb = []
            input_embeds = self.plm.get_input_embeddings()

            for i in range(len(label_dict)):
                label_emb.append(
                    input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
            label_emb = torch.stack(label_emb)
            print("------------------------------ label_emb.shape ------------------------------", label_emb.shape)
            ##

            if self.args.use_hier_mean:
                hier_mean_level = 0.25
                flat_slot2value = self.processor.flat_slot2value
                for depth_idx in range(self.args.depth - 2, -1, -1):

                    cd_labels = self.processor.depth2label[depth_idx]
                    for i in range(all_length):

                        if i in cd_labels:
                            fine_grained_label_emb = label_emb[list(flat_slot2value[i]), :].mean(dim=0)
                            label_emb[i] = fine_grained_label_emb * hier_mean_level + label_emb[i] * (
                                    1 - hier_mean_level)
                        else:
                            pass

            if "0.1.2" in openprompt.__path__[0]:
                self.verbalizer.head_last_layer.weight.data = label_emb
                self.verbalizer.head_last_layer.weight.data.requires_grad = True
            else:
                getattr(self.verbalizer.head.predictions, 'decoder').weight.data = label_emb
                getattr(self.verbalizer.head.predictions, 'decoder').weight.data.requires_grad = True

    def HierCRF_parameters_3(self):
        return [p for n, p in self.HierCRF.named_parameters()]


class BartForConditionalGenerationWithHierRCRF(nn.Module):

    def __init__(self, config, processor, args, tokenizer, device):
        super().__init__()
        self.args = args
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.processor = processor
        new_all_labels = ["_".join(label.split()) for label in self.processor.all_labels]

        if args.model_type == "bart":
            self.model = BartModel.from_pretrained(self.args.model_name_or_path)
            new_all_labels = [f"<{label}>" for label in new_all_labels]
            self.shift = 50265
        elif args.model_type == "T5":
            self.model = T5Model.from_pretrained(self.args.model_name_or_path)
            new_all_labels = [f"<<{label}>>" for label in new_all_labels]
            self.special_tokens_list = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

            new_tokens = self.tokenizer.add_tokens(
                self.special_tokens_list, special_tokens=True
            )
            self.shift = 32100 + new_tokens
        else:
            raise NotImplementedError
        # self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # self.lm_head = nn.Linear(self.config.d_model, self.model.shared.num_embeddings, bias=False)

        self.num_tags = len(self.processor.all_labels)
        self.verbalizer = nn.Linear(self.config.d_model, self.num_tags)

        start_transitions, end_transitions, transitions = None, None, None
        # if args.apply_transitions:
        #     start_transitions, end_transitions, transitions = self.init_transitions()
        self.HierCRF = HierCRF(self.num_tags, batch_first=True,
                               start_transitions=start_transitions,
                               end_transitions=end_transitions, transitions=transitions)
        self.to(self.device)
        self.new_label_emb = self.init_embedding()

        # increase the vocabulary of model and tokenizer

        self.new_tokens = self.tokenizer.add_tokens(new_all_labels)
        self.init_model_embedding()

        self.flag_use_ICRF = False
        self.flag_use_CE = False

    def init_model_embedding(self):
        input_embeds = self.model.get_input_embeddings()
        if self.args.model_type == "bart":
            new_input_embeds = torch.cat([input_embeds.weight, self.new_label_emb], dim=0)
        elif self.args.model_type == "T5":
            new_input_embeds = torch.cat([input_embeds.weight[:self.shift], self.new_label_emb], dim=0)
        embedding = nn.Embedding.from_pretrained(new_input_embeds, False, self.config.pad_token_id)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.set_input_embeddings(embedding)

    def init_embedding(self):
        if self.args.mean_verbalizer:
            print("using label emb for soft verbalizer")
            all_length = len(self.processor.all_labels)

            label_dict = dict({idx: v for idx, v in enumerate(self.processor.all_labels)})
            label_dict = {i: self.tokenizer.encode(v) for i, v in label_dict.items()}
            label_emb = []
            input_embeds = self.model.get_input_embeddings()

            for i in range(len(label_dict)):
                label_emb.append(
                    input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
            label_emb = torch.stack(label_emb)
            print("------------------------------ label_emb.shape ------------------------------", label_emb.shape)
            ##

            if self.args.use_hier_mean:
                hier_mean_level = 0.25
                flat_slot2value = self.processor.flat_slot2value
                for depth_idx in range(self.args.depth - 2, -1, -1):

                    cd_labels = self.processor.depth2label[depth_idx]
                    for i in range(all_length):

                        if i in cd_labels:
                            fine_grained_label_emb = label_emb[list(flat_slot2value[i]), :].mean(dim=0)
                            label_emb[i] = fine_grained_label_emb * hier_mean_level + label_emb[i] * (
                                    1 - hier_mean_level)
                        else:
                            pass

            self.verbalizer.weight.data = label_emb
            self.verbalizer.weight.data.requires_grad = True
            return label_emb

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            forward_type: str = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        # if forward_type == 'decode':
        #     print("input_ids", input_ids.shape)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.verbalizer(outputs[0])
        # lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        debug = False
        if debug:
            print(labels[0])
            print(input_ids[0][:40])

        masked_lm_loss = None
        if self.args.model_type == "bart":
            eos_indices = torch.nonzero(torch.eq(labels, 2))[0][1]
        elif self.args.model_type == "T5":
            eos_indices = torch.nonzero(torch.eq(labels, 1))[0][1]

        else:
            raise NotImplementedError
        lm_logits = lm_logits[:, :eos_indices, :]
        if forward_type == 'decode':
            return lm_logits
        if labels is not None:

            labels = labels[:, :eos_indices]
            labels = labels - self.shift
            labels = labels.to(lm_logits.device)
            if self.args.use_ICRF:
                if not self.flag_use_ICRF:
                    self.flag_use_ICRF = True
                    print("--------using ICRF--------")
                masked_lm_loss = -1 * self.HierCRF(lm_logits, labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                # print(lm_logits.shape)
                # print(labels.shape)
                # print("self.num_tags:", self.num_tags)
                if not self.flag_use_CE:
                    self.flag_use_CE = True
                    print("--------using CE--------")
                    print("lm_logits:", lm_logits.shape)
                    print("labels:", labels.shape)
                masked_lm_loss = loss_fct(lm_logits.contiguous().view(-1, self.num_tags), labels.contiguous().view(-1))

        # if labels is not None:
        #     labels = labels.to(lm_logits.device)
        #     loss_fct = nn.CrossEntropyLoss()
        #     masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        output = (lm_logits,) + outputs[1:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    def decode(self, batch):
        emissions = self(**batch, forward_type="decode")
        # batch_size * depth * num_tags(all_labels)

        return self.HierCRF.decode(emissions)

    def verbalizer_group_parameters(self, ):
        r"""Include the last layer's parameters
        """
        if isinstance(self.verbalizer, torch.nn.Linear):
            return [p for n, p in self.verbalizer.named_parameters()]

    def HierCRF_parameters(self):
        return [p for n, p in self.HierCRF.named_parameters()]

    def evaluate_decode(self, dataloader, processor, desc="Valid", mode=0, device="cuda:0", args=None, debug=False):
        self.eval()
        preds = []
        truth = []
        pbar = tqdm(dataloader, desc=desc + "_hierCRF")
        hier_mapping = processor.hier_mapping
        depth = len(hier_mapping) + 1

        for step, batch in enumerate(pbar):
            inputs = self._get_inputs_dict(batch)
            if self.args.use_ICRF:
                flat_preds = self.decode(inputs)
            else:
                logits = self.forward(**inputs, forward_type="decode")
                logits = torch.softmax(logits, dim=-1)
                flat_preds = torch.argmax(logits, dim=-1).cpu().tolist()

            if self.args.template_id == 6:
                if mode == 0:
                    flat_preds = [pred[-self.args.depth:] for pred in flat_preds]
                    for i in flat_preds:
                        i.reverse()
                elif mode == 1:
                    flat_preds = [pred[:self.args.depth] for pred in flat_preds]

            labels = inputs['labels']
            if self.args.model_type == "bart":
                eos_indices = torch.nonzero(torch.eq(labels, 2))[0][1]
            elif self.args.model_type == "T5":
                eos_indices = torch.nonzero(torch.eq(labels, 1))[0][1]
            else:
                raise NotImplementedError

            labels = labels[:, :eos_indices]
            labels = labels - self.shift
            labels = labels[:, -2:]
            labels = labels.tolist()
            for i in range(len(labels)):
                labels[i].reverse()
            batch_size = len(flat_preds)
            flat_preds = [list(set(flat_preds[i])) for i in range(batch_size)]

            for i in range(batch_size):
                preds.append(flat_preds[i])
            truth.extend(labels)

        if debug:
            print(preds[:200])
            print(truth[:200])
        label_dict = dict({idx: label for idx, label in enumerate(processor.all_labels)})
        assert len(preds) == len(truth)
        scores = compute_score(preds, truth, label_dict, debug)

        return scores

    def _get_inputs_dict(self, batch):
        device = self.device
        if self.args.model_type in ["marian"]:
            pad_token_id = self.tokenizer.pad_token_id
            source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100

            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "lm_labels": lm_labels.to(device),
            }
        elif self.args.model_type in ["blender", "bart", "T5", "blender-large"]:
            pad_token_id = self.tokenizer.pad_token_id
            source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]

            y_ids = y[:, :-1].contiguous()
            labels = y[:, 1:].clone()
            labels[y[:, 1:] == pad_token_id] = -100
            if True:
                masked_indices = torch.ones(labels.shape).bool()
                masked_indices = ~masked_indices
                masked_indices[labels != pad_token_id] = True
                masked_indices[labels != self.tokenizer.bos_token_id] = False
                masked_indices[labels != self.tokenizer.eos_token_id] = False
                labels[masked_indices] = labels[masked_indices] - self.shift
            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "labels": labels.to(device),
            }
        else:
            lm_labels = batch[1]
            lm_labels_masked = lm_labels.clone()
            lm_labels_masked[lm_labels_masked == self.decoder_tokenizer.pad_token_id] = -100

            inputs = {
                "input_ids": batch[0].to(device),
                "decoder_input_ids": lm_labels.to(device),
                "labels": lm_labels_masked.to(device),
            }

        return inputs
