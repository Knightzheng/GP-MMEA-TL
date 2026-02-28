import types
import torch
import transformers
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import pdb
import math
from .Tool_model import AutomaticWeightedLoss
from .MEAformer_tools import MultiModalEncoder
from .MEAformer_loss import CustomMultiLossLayer, icl_loss

from src.utils import pairwise_distances
import os.path as osp
import json


class MEAformer(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.kgs = kgs
        self.args = args
        self.img_features = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()
        self.img_mask = None
        if "img_mask" in kgs and kgs["img_mask"] is not None:
            self.img_mask = torch.FloatTensor(kgs["img_mask"]).cuda()
        self.input_idx = kgs["input_idx"].cuda()
        self.adj = kgs["adj"].cuda()
        self.rel_features = torch.Tensor(kgs["rel_features"]).cuda()
        self.att_features = torch.Tensor(kgs["att_features"]).cuda()
        self.name_features = None
        self.char_features = None
        if kgs["name_features"] is not None:
            self.name_features = kgs["name_features"].cuda()
            self.char_features = kgs["char_features"].cuda()

        img_dim = self._get_img_dim(kgs)

        char_dim = kgs["char_features"].shape[1] if self.char_features is not None else 100

        self.multimodal_encoder = MultiModalEncoder(args=self.args,
                                                    ent_num=kgs["ent_num"],
                                                    img_feature_dim=img_dim,
                                                    char_feature_dim=char_dim,
                                                    use_project_head=self.args.use_project_head,
                                                    attr_input_dim=kgs["att_features"].shape[1])

        self.multi_loss_layer = CustomMultiLossLayer(loss_num=6)  # 6
        self.criterion_cl = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        self.criterion_cl_joint = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2, replay=self.args.replay, neg_cross_kg=self.args.neg_cross_kg)

        tmp = -1 * torch.ones(self.input_idx.shape[0], dtype=torch.int64).cuda()
        self.replay_matrix = torch.stack([self.input_idx, tmp], dim=1).cuda()
        self.replay_ready = 0
        self.idx_one = torch.ones(self.args.batch_size, dtype=torch.int64).cuda()
        self.idx_double = torch.cat([self.idx_one, self.idx_one]).cuda()
        self.last_num = 1000000000000
        # self.idx_one = np.ones(self.args.batch_size, dtype=np.int64)

    def _to_cuda_batch(self, batch, device):
        if not torch.is_tensor(batch):
            return torch.as_tensor(batch, dtype=torch.long, device=device)
        return batch.to(device=device, dtype=torch.long)

    def _domain_align_loss(self, joint_emb, batch):
        if not getattr(self.args, "use_domain_align", 0):
            return None
        if batch is None:
            return None
        batch = self._to_cuda_batch(batch, joint_emb.device)
        if batch.numel() == 0:
            return None

        left_emb = joint_emb[batch[:, 0]]
        right_emb = joint_emb[batch[:, 1]]
        return F.mse_loss(left_emb, right_emb)

    def _missing_aware_img_align_loss(self, img_emb, batch):
        if not getattr(self.args, "use_missing_gate", 0):
            return None
        if img_emb is None or self.img_mask is None or batch is None:
            return None
        batch = self._to_cuda_batch(batch, img_emb.device)
        left = batch[:, 0]
        right = batch[:, 1]
        valid = (self.img_mask[left] > 0.5) & (self.img_mask[right] > 0.5)
        if torch.sum(valid) == 0:
            return torch.tensor(0.0, device=img_emb.device)
        return F.mse_loss(img_emb[left[valid]], img_emb[right[valid]])

    def _source_select_loss(self, modal_losses):
        if not getattr(self.args, "use_source_select", 0):
            return None, {}
        active = [(k, v) for k, v in modal_losses.items() if v is not None]
        if len(active) == 0:
            return None, {}
        values = torch.stack([v for _, v in active])
        temp = max(float(getattr(self.args, "source_select_temp", 1.0)), 1e-6)
        weights = torch.softmax(-values.detach() / temp, dim=0)
        selected_loss = torch.sum(weights * values)
        weight_dict = {f"src_w_{name}": weights[i].item() for i, (name, _) in enumerate(active)}
        return selected_loss, weight_dict

    def forward(self, batch):
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states = self.joint_emb_generat(only_joint=False)
        gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, joint_emb_hid = self.generate_hidden_emb(hidden_states)
        batch = self._to_cuda_batch(batch, joint_emb.device)
        if self.args.replay:
            all_ent_batch = torch.cat([batch[:, 0], batch[:, 1]])
            if not self.replay_ready:
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch)
            else:
                neg_l = self.replay_matrix[batch[:, 0], self.idx_one[:batch.shape[0]]]
                neg_r = self.replay_matrix[batch[:, 1], self.idx_one[:batch.shape[0]]]
                neg_l_set = set(neg_l.tolist())
                neg_r_set = set(neg_r.tolist())
                all_ent_set = set(all_ent_batch.tolist())
                neg_l_list = list(neg_l_set - all_ent_set)
                neg_r_list = list(neg_r_set - all_ent_set)
                neg_l_ipt = torch.tensor(neg_l_list, dtype=torch.int64).cuda()
                neg_r_ipt = torch.tensor(neg_r_list, dtype=torch.int64).cuda()
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch, neg_l_ipt, neg_r_ipt)

            index = (
                all_ent_batch,
                self.idx_double[:batch.shape[0] * 2],
            )
            new_value = torch.cat([l_neg, r_neg]).cuda()

            self.replay_matrix = self.replay_matrix.index_put(index, new_value)
            if self.replay_ready == 0:
                num = torch.sum(self.replay_matrix < 0)
                if num == self.last_num:
                    self.replay_ready = 1
                    print("-----------------------------------------")
                    print("begin replay!")
                    print("-----------------------------------------")
                else:
                    self.last_num = num
        else:
            loss_joi = self.criterion_cl_joint(joint_emb, batch)

        in_loss, in_info = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch)
        out_loss, out_info = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, batch)

        loss_all = loss_joi + in_loss + out_loss
        domain_align_loss = self._domain_align_loss(joint_emb, batch)
        if domain_align_loss is not None and self.args.domain_align_weight > 0:
            loss_all = loss_all + self.args.domain_align_weight * domain_align_loss
        missing_align_loss = self._missing_aware_img_align_loss(img_emb, batch)
        if missing_align_loss is not None and self.args.missing_align_weight > 0:
            loss_all = loss_all + self.args.missing_align_weight * missing_align_loss

        source_select_loss = in_info["source_loss"] + out_info["source_loss"]

        loss_dic = {
            "joint_Intra_modal": loss_joi.item(),
            "Intra_modal": in_loss.item(),
            "domain_align": domain_align_loss.item() if domain_align_loss is not None else 0.0,
            "missing_align": missing_align_loss.item() if missing_align_loss is not None else 0.0,
            "source_select": source_select_loss.item() if torch.is_tensor(source_select_loss) else 0.0,
        }
        for k, v in in_info.get("source_weights", {}).items():
            loss_dic[f"in_{k}"] = v
        for k, v in out_info.get("source_weights", {}).items():
            loss_dic[f"out_{k}"] = v
        output = {"loss_dic": loss_dic, "emb": joint_emb}
        return loss_all, output

    def generate_hidden_emb(self, hidden):
        gph_emb = F.normalize(hidden[:, 0, :].squeeze(1))
        rel_emb = F.normalize(hidden[:, 1, :].squeeze(1))
        att_emb = F.normalize(hidden[:, 2, :].squeeze(1))
        img_emb = F.normalize(hidden[:, 3, :].squeeze(1))
        if hidden.shape[1] >= 6:
            name_emb = F.normalize(hidden[:, 4, :].squeeze(1))
            char_emb = F.normalize(hidden[:, 5, :].squeeze(1))
            joint_emb = torch.cat([gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb], dim=1)
        else:
            name_emb, char_emb = None, None
            joint_emb = torch.cat([gph_emb, rel_emb, att_emb, img_emb], dim=1)

        return gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, joint_emb

    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill):
        modal_losses = {
            "gcn": self.criterion_cl(gph_emb, train_ill) if gph_emb is not None else None,
            "rel": self.criterion_cl(rel_emb, train_ill) if rel_emb is not None else None,
            "att": self.criterion_cl(att_emb, train_ill) if att_emb is not None else None,
            "img": self.criterion_cl(img_emb, train_ill) if img_emb is not None else None,
            "name": self.criterion_cl(name_emb, train_ill) if name_emb is not None else None,
            "char": self.criterion_cl(char_emb, train_ill) if char_emb is not None else None,
        }

        zero = torch.tensor(0.0, device=self.img_features.device)
        total_loss = self.multi_loss_layer([
            modal_losses["gcn"] if modal_losses["gcn"] is not None else zero,
            modal_losses["rel"] if modal_losses["rel"] is not None else zero,
            modal_losses["att"] if modal_losses["att"] is not None else zero,
            modal_losses["img"] if modal_losses["img"] is not None else zero,
            modal_losses["name"] if modal_losses["name"] is not None else zero,
            modal_losses["char"] if modal_losses["char"] is not None else zero,
        ])

        source_loss, source_weights = self._source_select_loss(modal_losses)
        if source_loss is None:
            source_loss = zero
        elif self.args.source_select_weight > 0:
            total_loss = total_loss + self.args.source_select_weight * source_loss

        return total_loss, {"source_loss": source_loss, "source_weights": source_weights}

    # --------- necessary ---------------

    def joint_emb_generat(self, only_joint=True):
        gph_emb, img_emb, rel_emb, att_emb, \
            name_emb, char_emb, joint_emb, hidden_states, weight_norm = self.multimodal_encoder(self.input_idx,
                                                                                                self.adj,
                                                                                                self.img_features,
                                                                                                self.rel_features,
                                                                                                self.att_features,
                                                                                                self.name_features,
                                                                                                self.char_features)
        if only_joint:
            return joint_emb, weight_norm
        else:
            return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states

    # --------- share ---------------

    def _get_img_dim(self, kgs):
        if isinstance(kgs["images_list"], list):
            img_dim = kgs["images_list"][0].shape[1]
        elif isinstance(kgs["images_list"], np.ndarray) or torch.is_tensor(kgs["images_list"]):
            img_dim = kgs["images_list"].shape[1]
        return img_dim

    def Iter_new_links(self, epoch, left_non_train, final_emb, right_non_train, new_links=[]):
        if len(left_non_train) == 0 or len(right_non_train) == 0:
            return new_links
        distance_list = []
        for i in np.arange(0, len(left_non_train), 1000):
            d = pairwise_distances(final_emb[left_non_train[i:i + 1000]], final_emb[right_non_train])
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        if (epoch + 1) % (self.args.semi_learn_step * 5) == self.args.semi_learn_step:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if preds_r[p] == i]
        else:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if (preds_r[p] == i) and ((left_non_train[i], right_non_train[p]) in new_links)]

        return new_links

    def data_refresh(self, logger, train_ill, test_ill_, left_non_train, right_non_train, new_links=[]):
        if len(new_links) != 0 and (len(left_non_train) != 0 and len(right_non_train) != 0):
            new_links_select = new_links
            train_ill = np.vstack((train_ill, np.array(new_links_select)))
            num_true = len([nl for nl in new_links_select if nl in test_ill_])
            # remove from left/right_non_train
            for nl in new_links_select:
                left_non_train.remove(nl[0])
                right_non_train.remove(nl[1])

            if self.args.rank == 0:
                logger.info(f"#new_links_select:{len(new_links_select)}")
                logger.info(f"train_ill.shape:{train_ill.shape}")
                logger.info(f"#true_links: {num_true}")
                logger.info(f"true link ratio: {(100 * num_true / len(new_links_select)):.1f}%")
                logger.info(f"#entity not in train set: {len(left_non_train)} (left) {len(right_non_train)} (right)")

            new_links = []
        else:
            logger.info("len(new_links) is 0")

        return left_non_train, right_non_train, train_ill, new_links
