import copy
import logging
from collections import Counter
from typing import Dict, Iterable, Union

import numpy as np
import torch
import torch.nn as nn

from alpaca.uncertainty_estimator.masks import build_mask
from transformers import ElectraForSequenceClassification
from transformers.activations import get_activation


log = logging.getLogger(__name__)


class DropoutMC(torch.nn.Module):
    def __init__(self, p: float, activate=False):
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.dropout(x, self.p, training=self.training or self.activate)


class LockedDropoutMC(DropoutMC):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """

    def __init__(self, p: float, activate: bool = False, batch_first: bool = True):
        super().__init__(p, activate)
        self.batch_first = batch_first

    def forward(self, x):
        if self.training:
            self.activate = True
        # if not self.training or not self.p:
        if not self.activate or not self.p:
            return x

        if not self.batch_first:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.p)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)

        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.p)
        mask = mask.expand_as(x)
        return mask * x


class WordDropoutMC(DropoutMC):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def forward(self, x):
        if self.training:
            self.activate = True

        # if not self.training or not self.p:
        if not self.activate or not self.p:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.p)

        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x


MC_DROPOUT_SUBSTITUTES = {
    "Dropout": DropoutMC,
    "LockedDropout": LockedDropoutMC,
    "WordDropout": WordDropoutMC,
}


def convert_to_mc_dropout(model: torch.nn.Module, substitution_dict: Dict[str, torch.nn.Module] = None):
    for i, layer in enumerate(list(model.children())):
        proba_field_name = "dropout_rate" if "flair" in str(type(layer)) else "p"
        module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        if layer_name in substitution_dict.keys():
            model._modules[module_name] = substitution_dict[layer_name](
                p=getattr(layer, proba_field_name), activate=False
            )
        else:
            convert_to_mc_dropout(model=layer, substitution_dict=substitution_dict)


def activate_mc_dropout(model: torch.nn.Module, activate: bool, random: float = 0.0, verbose: bool = False):
    for layer in model.children():
        if isinstance(layer, DropoutMC):
            if verbose:
                print(layer)
                print(f"Current DO state: {layer.activate}")
                print(f"Switching state to: {activate}")
            layer.activate = activate
            if activate and random:
                layer.p = random
            if not activate:
                layer.p = layer.p_init
        else:
            activate_mc_dropout(model=layer, activate=activate, random=random, verbose=verbose)


############################################################################################################
# Common functions: dropout_mc.py, ue_scores.py, ue_variation_ratio.py
############################################################################################################


def data_uncertainty(preds, ue="vanilla"):
    """
    Input:
        preds: B X T X C
    Output:
        scores: B
    """
    if ue == "vanilla":
        token_score = 1 - torch.max(preds, dim=-1)[0]  # B X T
    elif ue == "entropy":
        token_score = torch.sum(-preds * torch.log(torch.clip(preds, 1e-8, 1)), axis=-1)  # B X T
    else:
        raise ValueError("Unknown uncertainty estimation method.")

    return torch.mean(token_score, dim=-1)  # B


def entropy(x):
    return np.sum(-x * np.log(np.clip(x, 1e-8, 1)), axis=-1)


def mean_entropy(sampled_probabilities):
    return entropy(np.mean(sampled_probabilities, axis=1))


def var_ratio(sampled_probabilities):
    top_classes = np.argmax(sampled_probabilities, axis=-1)
    # count how many time repeats the strongest class
    mode_count = lambda preds: np.max(np.bincount(preds))
    modes = [mode_count(point) for point in top_classes]
    ue = 1.0 - np.array(modes) / sampled_probabilities.shape[1]
    return ue


def sampled_max_prob(sampled_probabilities):
    """Computes the max probability for a set of samples.
    Args:
        sampled_probabilities: A numpy array of K forward passes, where each pass contains an array of batch size B X length T X class C.
    Returns:
        max_prob: A numpy array of batch size B X length T.

    def sampled_max_prob(sampled_probabilities):
        mean_probabilities = np.mean(sampled_probabilities, axis=1)
        top_probabilities = np.max(mean_probabilities, axis=-1)
        return 1 - top_probabilities
    """
    if not isinstance(sampled_probabilities, np.ndarray):
        sampled_probabilities = np.array(sampled_probabilities)

    # Compute the mean probability over the K forward passes.
    max_prob = np.max(np.mean(sampled_probabilities, axis=0), axis=-1)  # K X B X T X C -> B X T X C -> B X T

    return np.mean(1 - max_prob, axis=-1)  # B


def probability_variance(sampled_probabilities):
    """Computes the probability variance for a set of samples.
    Args:
        sampled_probabilities: A numpy array of K forward passes, where each pass contains an array of batch size B X length T X class C.
    Returns:
        variance: A numpy array of batch size B X length T.

    def probability_variance(sampled_probabilities):
        mean_probabilities = np.expand_dims(mean_probabilities, axis=1)
        return ((sampled_probabilities - mean_probabilities) ** 2).mean(1).sum(-1)
    """
    if not isinstance(sampled_probabilities, np.ndarray):
        sampled_probabilities = np.array(sampled_probabilities)

    # Compute the mean probability over the K forward passes.
    mean_probabilities = np.expand_dims(
        np.mean(sampled_probabilities, axis=0), axis=0
    )  # K X B X T X C -> 1 X B X T X C
    variance = np.mean(np.power(sampled_probabilities - mean_probabilities, 2), axis=0)  # B X T X C
    variance = np.mean(np.sum(variance, axis=-1), axis=-1)  # B X T -> B

    return variance  # B


def bald(sampled_probabilities):
    """Computes the BALD score for a set of samples.
    Args:
        sampled_probabilities: A numpy array of K forward passes, where each pass contains an array of batch size B X length T X class C.
    Returns:
        bald: A numpy array of batch size B X length T.
    """
    if not isinstance(sampled_probabilities, np.ndarray):
        sampled_probabilities = np.array(sampled_probabilities)

    # Compute the mean probability over the K forward passes.
    predictive_entropy = entropy(np.mean(sampled_probabilities, axis=0))  # K X B X T X C -> B X T X C -> B X T
    expected_entropy = np.mean(entropy(sampled_probabilities), axis=0)  # K X B X T X C -> K X B X T -> B X T

    return np.mean(predictive_entropy - expected_entropy, axis=-1)  # B


def find_most_common(row: Iterable[str], mode: Union["elem", "count"]):
    """
    Given iterable of words, return either most common element or its count
    """
    if mode == "elem":
        return Counter(row).most_common(1)[0][0]
    elif mode == "count":
        return Counter(row).most_common(1)[0][1]


def ue_variation_ratio(answers):
    answers = [np.array(e, dtype=object) for e in answers]
    answers = np.stack(answers, -1)

    scores = 1.0 - np.array([find_most_common(ans, "count") / answers.shape[1] for ans in answers])
    return scores


def entities2dict(entities, queryid, ent_dict):
    """
    We build ent_dict iterately for each instance, each item contains:
        key: the query entity index tuple,
        values: a dict including the query entity tag, query entity index, and related entity info.
    Outputs:
        ent_dict (dict): {
            record_idx1: {"entity_group": Tag1, "word": word1, "related_ent": {idx1: (tag1, word1), ...}},
            record_idx2: {"entity_group": Tag2, "word": word2, "related_ent": {idx2: (tag2, word1), ...}},
            ...
        }
    """
    related_ent = dict()
    ent_record = None
    for entity in entities:
        tag, index, word = entity["entity_group"], sorted(entity["index"]), entity["word"].strip()
        if index[0] == queryid:  # query entity
            ent_dict[tuple(index)]["entity_group"] = tag
            ent_dict[tuple(index)]["word"] = word
            ent_record = tuple(index)
        else:  # other related entities
            related_ent[tuple(index)] = (tag, word)

    if ent_record is not None:  # if query entity exist we also record its related entities
        ent_dict[ent_record]["related_ent"] = related_ent
    else:
        if related_ent:  # no query entity but predict other related entities
            ent_dict[tuple([queryid])]["entity_group"] = "None"
            ent_dict[tuple([queryid])]["word"] = ""
            ent_dict[tuple([queryid])]["related_ent"] = related_ent


def merge_ent_dict(ent_dict, sent_ents):
    """
    We use the ent_dict to interately extract all triplets in the form:
        {"ent1": idx1, "ent1_tag": tag1, "ent2": idx2, "ent2_tag": tag2}.
    Each triplet is then added to sent_ents.
    """
    for ent1, items in ent_dict.items():
        ent1_tag, ent1_word = items["entity_group"], items["word"]
        if not items["related_ent"]:  # no related entities (empty dict)
            sent_ents.append(
                {
                    "ent1": ent1_word,
                    "ent1_tag": ent1_tag,
                    "ent2": "",
                    "ent2_tag": "None",
                }
            )
        else:  # iterately append each related entity triplet
            for ent2, (ent2_tag, ent2_word) in items["related_ent"].items():
                sent_ents.append(
                    {
                        "ent1": ent1_word,
                        "ent1_tag": ent1_tag,
                        "ent2": ent2_word,
                        "ent2_tag": ent2_tag,
                    }
                )


def common_cal(preds, labels):
    """
    Both preds and labels are a list of triplets (dicts).
    """
    n_hyp = len(preds)
    n_ref = len(labels)

    false_tag = 0
    re_fn = 0
    re_fp = 0
    re_tag_f = 0
    re_mention_f = 0
    ent_mention_f = 0
    ent_tag_f = 0

    # consider ent1_tag
    intersection_tag = [ent for ent in preds if ent in labels]
    tp_tag = len(intersection_tag)

    for ent1 in preds:
        for ent2 in labels:
            # if ent1 != ent2 and ent1['ent1'][0] == ent2['ent1'][0]: #  we define a corresponding pair
            if ent1 != ent2 and ent1["ent1"] == ent2["ent1"]:  #  we define a corresponding pair
                false_tag += 1
                if ent1["ent1"] != ent2["ent1"]:  # incorrect entity mention prediction
                    ent_mention_f += 1
                if ent1["ent1_tag"] != ent2["ent1_tag"]:
                    if ent1["ent1_tag"] != "None" and ent2["ent1_tag"] != "None":
                        ent_tag_f += 1
                if ent1["ent2"] != ent2["ent2"]:
                    if ent1["ent2"] != tuple() and ent2["ent2"] != tuple():
                        re_mention_f += 1
                if ent1["ent2_tag"] != ent2["ent2_tag"]:  # incorrect relation prediction
                    if ent1["ent2_tag"] == "None" and ent2["ent2_tag"] != "None":  # relation false negative
                        re_fn += 1
                    elif ent1["ent2_tag"] != "None" and ent2["ent2_tag"] == "None":  # relation false positive
                        re_fp += 1
                    else:
                        re_tag_f += 1

    # not considering ingent1_tag
    removed_keys = [
        "ent1_tag",
    ]
    for rm_key in removed_keys:
        for ent1 in preds:
            ent1.pop(rm_key, None)
        for ent2 in labels:
            ent2.pop(rm_key, None)

    intersection_notag = [ent for ent in preds if ent in labels]
    tp_notag = len(intersection_notag)

    return tp_notag, tp_tag, n_hyp, n_ref, false_tag, ent_mention_f, ent_tag_f, re_mention_f, re_fn, re_fp, re_tag_f


class ElectraClassificationHeadCustom(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, other):
        super().__init__()
        self.dropout1 = other.dropout
        self.dense = other.dense
        self.dropout2 = copy.deepcopy(other.dropout)
        self.out_proj = other.out_proj

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout1(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout2(x)
        x = self.out_proj(x)
        return x


class DropoutDPP(DropoutMC):
    dropout_id = -1

    def __init__(
        self,
        p: float,
        activate=False,
        mask_name="dpp",
        max_n=100,
        max_frac=0.4,
        coef=1.0,
    ):
        super().__init__(p=p, activate=activate)

        self.mask = build_mask(mask_name)
        self.reset_mask = False
        self.max_n = max_n
        self.max_frac = max_frac
        self.coef = coef

        self.curr_dropout_id = DropoutDPP.update()
        log.debug(f"Dropout id: {self.curr_dropout_id}")

    @classmethod
    def update(cls):
        cls.dropout_id += 1
        return cls.dropout_id

    def calc_mask(self, x: torch.Tensor):
        return self.mask(x, dropout_rate=self.p, layer_num=self.curr_dropout_id).float()

    def get_mask(self, x: torch.Tensor):
        return self.mask(x, dropout_rate=self.p, layer_num=self.curr_dropout_id).float()

    def calc_non_zero_neurons(self, sum_mask):
        frac_nonzero = (sum_mask != 0).sum(axis=-1).item() / sum_mask.shape[-1]
        return frac_nonzero

    def forward(self, x: torch.Tensor):
        if self.training:
            return torch.nn.functional.dropout(x, self.p, training=True)
        else:
            if not self.activate:
                return x

            sum_mask = self.get_mask(x)

            norm = 1.0
            i = 1
            frac_nonzero = self.calc_non_zero_neurons(sum_mask)
            # print('==========Non zero neurons:', frac_nonzero, 'iter:', i, 'id:', self.curr_dropout_id, '******************')
            # while i < 30:
            while i < self.max_n and frac_nonzero < self.max_frac:
                # while frac_nonzero < self.max_frac:
                mask = self.get_mask(x)

                # sum_mask = self.coef * sum_mask + mask
                sum_mask += mask
                i += 1
                # norm = self.coef * norm + 1

                frac_nonzero = self.calc_non_zero_neurons(sum_mask)
                # print('==========Non zero neurons:', frac_nonzero, 'iter:', i, '******************')

            # res = x * sum_mask / norm
            print("Number of masks:", i)
            res = x * sum_mask / i
            return res


def get_last_dropout(model):
    if isinstance(model, ElectraForSequenceClassification):
        if isinstance(model.classifier, ElectraClassificationHeadCustom):
            return model.classifier.dropout2
        else:
            return model.classifier.dropout
    else:
        return model.dropout


def set_last_dropout(model, dropout):
    if isinstance(model, ElectraForSequenceClassification):
        if isinstance(model.classifier, ElectraClassificationHeadCustom):
            model.classifier.dropout2 = dropout
        else:
            model.classifier.dropout
    else:
        model.dropout = dropout


def convert_dropouts(model, ue_args):
    if ue_args.dropout_type == "MC":
        dropout_ctor = lambda p, activate: DropoutMC(p=ue_args.inference_prob, activate=False)
    elif ue_args.dropout_type == "DPP":

        def dropout_ctor(p, activate):
            return DropoutDPP(
                p=p,
                activate=activate,
                max_n=ue_args.dropout.max_n,
                max_frac=ue_args.dropout.max_frac,
                mask_name=ue_args.dropout.mask_name,
            )

    else:
        raise ValueError(f"Wrong dropout type: {ue_args.dropout_type}")

    if ue_args.dropout_subs == "last":
        set_last_dropout(model, dropout_ctor(p=ue_args.inference_prob, activate=False))

    elif ue_args.dropout_subs == "all":
        # convert_to_mc_dropout(model, {'Dropout': dropout_ctor})
        convert_to_mc_dropout(model.electra.encoder, {"Dropout": dropout_ctor})
    else:
        raise ValueError(f"Wrong ue args {ue_args.dropout_subs}")


def calculate_dropouts(model):
    res = 0
    for i, layer in enumerate(list(model.children())):
        module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        if layer_name == "Dropout":
            res += 1
        else:
            res += calculate_dropouts(model=layer)

    return res


def freeze_all_dpp_dropouts(model, freeze):
    for layer in model.children():
        if isinstance(layer, DropoutDPP):
            if freeze:
                layer.mask.freeze(dry_run=True)
            else:
                layer.mask.unfreeze(dry_run=True)
        else:
            freeze_all_dpp_dropouts(model=layer, freeze=freeze)
