import torch
import random
import json
import numpy as np
import pdb
import torch.distributed as dist
import os
import os.path as osp
from collections import Counter
import pickle
import torch.nn.functional as F
from transformers import BertTokenizer
import torch.distributed
from tqdm import tqdm

from .utils import get_topk_indices, get_adjr


class EADataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Collator_base(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, batch):
        # pdb.set_trace()

        return np.array(batch)


def load_data(logger, args):
    assert args.data_choice in ["DWY", "DBP15K", "FBYG15K", "FBDB15K"]
    if args.data_choice in ["DWY", "DBP15K", "FBYG15K", "FBDB15K"]:
        KGs, non_train, train_ill, test_ill, eval_ill, test_ill_ = load_eva_data(logger, args)

    elif args.data_choice in ["FBYG15K_attr", "FBDB15K_attr"]:
        pass

    return KGs, non_train, train_ill, test_ill, eval_ill, test_ill_


def load_eva_data(logger, args):
    file_dir = osp.join(args.data_path, args.data_choice, args.data_split)
    lang_list = [1, 2]
    ent2id_dict, ills, triples, r_hs, r_ts, ids = read_raw_data(file_dir, lang_list)
    e1 = os.path.join(file_dir, 'ent_ids_1')
    e2 = os.path.join(file_dir, 'ent_ids_2')
    left_ents = get_ids(e1)
    right_ents = get_ids(e2)
    ENT_NUM = len(ent2id_dict)
    REL_NUM = len(r_hs)
    np.random.shuffle(ills)
    if "V1" in file_dir:
        split = "norm"
        img_vec_path = osp.join(args.data_path, "pkls/dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl")
    elif "V2" in file_dir:
        split = "dense"
        img_vec_path = osp.join(args.data_path, "pkls/dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl")
    elif "FB" in file_dir:
        img_vec_path = osp.join(args.data_path, f"pkls/{args.data_choice}_id_img_feature_dict.pkl")
    else:
        # fr_en
        split = file_dir.split("/")[-1]
        img_vec_path = osp.join(args.data_path, "pkls", args.data_split + "_GA_id_img_feature_dict.pkl")

    assert osp.exists(img_vec_path)
    img_features, img_mask = load_img(logger, ENT_NUM, img_vec_path, args=args)
    logger.info(f"image feature shape:{img_features.shape}")

    if args.word_embedding == "glove":
        word2vec_path = os.path.join(args.data_path, "embedding", "glove.6B.300d.txt")
    elif args.word_embedding == 'bert':
        pass
    else:
        raise Exception("error word embedding")

    name_features = None
    char_features = None
    if args.data_choice == "DBP15K" and (args.w_name or args.w_char):

        assert osp.exists(word2vec_path)
        ent_vec, char_features = load_word_char_features(ENT_NUM, word2vec_path, args, logger)
        name_features = F.normalize(torch.Tensor(ent_vec))
        char_features = F.normalize(torch.Tensor(char_features))
        logger.info(f"name feature shape:{name_features.shape}")
        logger.info(f"char feature shape:{char_features.shape}")

    if args.unsup:
        mode = args.unsup_mode
        if mode == "char":
            input_features = char_features
        elif mode == "name":
            input_features = name_features
        else:
            input_features = F.normalize(torch.Tensor(img_features))

        train_ill = visual_pivot_induction(args, left_ents, right_ents, input_features, ills, logger)
    else:
        train_ill = np.array(ills[:int(len(ills) // 1 * args.data_rate)], dtype=np.int32)

    test_ill_ = ills[int(len(ills) // 1 * args.data_rate):]
    test_ill = np.array(test_ill_, dtype=np.int32)

    test_left = torch.LongTensor(test_ill[:, 0].squeeze())
    test_right = torch.LongTensor(test_ill[:, 1].squeeze())

    left_non_train = list(set(left_ents) - set(train_ill[:, 0].tolist()))

    right_non_train = list(set(right_ents) - set(train_ill[:, 1].tolist()))

    logger.info(f"#left entity : {len(left_ents)}, #right entity: {len(right_ents)}")
    logger.info(f"#left entity not in train set: {len(left_non_train)}, #right entity not in train set: {len(right_non_train)}")

    rel_features = load_relation(ENT_NUM, triples, 1000)
    logger.info(f"relation feature shape:{rel_features.shape}")
    a1 = os.path.join(file_dir, 'training_attrs_1')
    a2 = os.path.join(file_dir, 'training_attrs_2')
    att_features = load_attr([a1, a2], ENT_NUM, ent2id_dict, 1000)  # attr
    logger.info(f"attribute feature shape:{att_features.shape}")

    logger.info("-----dataset summary-----")
    logger.info(f"dataset:\t\t {file_dir}")
    logger.info(f"triple num:\t {len(triples)}")
    logger.info(f"entity num:\t {ENT_NUM}")
    logger.info(f"relation num:\t {REL_NUM}")
    logger.info(f"train ill num:\t {train_ill.shape[0]} \t test ill num:\t {test_ill.shape[0]}")
    logger.info("-------------------------")

    eval_ill = None
    input_idx = torch.LongTensor(np.arange(ENT_NUM))
    adj = get_adjr(ENT_NUM, triples, norm=True)
    # pdb.set_trace()
    train_ill = EADataset(train_ill)
    test_ill = EADataset(test_ill)

    return {
        'ent_num': ENT_NUM,
        'rel_num': REL_NUM,
        'images_list': img_features,
        'img_mask': img_mask,
        'rel_features': rel_features,
        'att_features': att_features,
        'name_features': name_features,
        'char_features': char_features,
        'input_idx': input_idx,
        'adj': adj
    }, {"left": left_non_train, "right": right_non_train}, train_ill, test_ill, eval_ill, test_ill_


def load_word2vec(path, dim=300):
    """
    glove or fasttext embedding
    """
    # print('\n', path)
    word2vec = dict()
    err_num = 0
    err_list = []

    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines(), desc="load word embedding"):
            line = line.strip('\n').split(' ')
            if len(line) != dim + 1:
                continue
            try:
                v = np.array(list(map(float, line[1:])), dtype=np.float64)
                word2vec[line[0].lower()] = v
            except:
                err_num += 1
                err_list.append(line[0])
                continue
    file.close()
    print("err list ", err_list)
    print("err num ", err_num)
    return word2vec


def load_char_bigram(path):
    """
    character bigrams of translated entity names
    """
    # load the translated entity names
    ent_names = json.load(open(path, "r"))
    # generate the bigram dictionary
    char2id = {}
    count = 0
    for _, name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word) - 1):
                if word[idx:idx + 2] not in char2id:
                    char2id[word[idx:idx + 2]] = count
                    count += 1
    return ent_names, char2id


def load_word_char_features(node_size, word2vec_path, args, logger):
    """
    node_size : ent num
    """
    name_path = os.path.join(args.data_path, "DBP15K", "translated_ent_name", "dbp_" + args.data_split + ".json")
    assert osp.exists(name_path)
    save_path_name = os.path.join(args.data_path, "embedding", f"dbp_{args.data_split}_name.pkl")
    save_path_char = os.path.join(args.data_path, "embedding", f"dbp_{args.data_split}_char.pkl")
    if osp.exists(save_path_name) and osp.exists(save_path_char):
        logger.info(f"load entity name emb from {save_path_name} ... ")
        ent_vec = pickle.load(open(save_path_name, "rb"))
        logger.info(f"load entity char emb from {save_path_char} ... ")
        char_vec = pickle.load(open(save_path_char, "rb"))
        return ent_vec, char_vec

    word_vecs = load_word2vec(word2vec_path)
    ent_names, char2id = load_char_bigram(name_path)

    # generate the word-level features and char-level features

    ent_vec = np.zeros((node_size, 300))
    char_vec = np.zeros((node_size, len(char2id)))
    for i, name in ent_names:
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
            for idx in range(len(word) - 1):
                char_vec[i, char2id[word[idx:idx + 2]]] += 1
        if k:
            ent_vec[i] /= k
        else:
            ent_vec[i] = np.random.random(300) - 0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(char2id)) - 0.5
        ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])
        char_vec[i] = char_vec[i] / np.linalg.norm(char_vec[i])

    with open(save_path_name, 'wb') as f:
        pickle.dump(ent_vec, f)
    with open(save_path_char, 'wb') as f:
        pickle.dump(char_vec, f)
    logger.info("save entity emb done. ")
    return ent_vec, char_vec


def visual_pivot_induction(args, left_ents, right_ents, img_features, ills, logger):
    def build_legacy_visual_links(sim_mat, topk, min_sim, dyn_q):
        cand_k = max(1, min(int(topk * 100), int(sim_mat.numel())))
        two_d_indices = get_topk_indices(sim_mat, cand_k)
        cand_sims = sim_mat[two_d_indices[:, 0], two_d_indices[:, 1]].detach().cpu().numpy()
        dyn_thr = -1.0
        if dyn_q > 0.0 and cand_sims.size > 0:
            dyn_thr = float(np.quantile(cand_sims, dyn_q))
        final_thr = max(min_sim, dyn_thr)

        visual_links = []
        used_inds = set()
        passed_threshold = 0
        for ind, sim in zip(two_d_indices, cand_sims):
            if sim < final_thr:
                continue
            passed_threshold += 1
            left_id = left_ents[ind[0]]
            right_id = right_ents[ind[1]]
            if left_id in used_inds or right_id in used_inds:
                continue
            used_inds.add(left_id)
            used_inds.add(right_id)
            visual_links.append((left_id, right_id))
            if len(visual_links) == topk:
                break

        fallback_added = 0
        if len(visual_links) < topk:
            for ind in two_d_indices:
                left_id = left_ents[ind[0]]
                right_id = right_ents[ind[1]]
                if left_id in used_inds or right_id in used_inds:
                    continue
                used_inds.add(left_id)
                used_inds.add(right_id)
                visual_links.append((left_id, right_id))
                fallback_added += 1
                if len(visual_links) == topk:
                    break

        return {
            "visual_links": visual_links,
            "candidate_count": cand_k,
            "dyn_thr": dyn_thr,
            "final_thr": final_thr,
            "passed_threshold": passed_threshold,
            "fallback_added": fallback_added,
            "filter_mode": "legacy_global_topk",
            "quality_thr": final_thr,
        }

    def build_bipartite_visual_links(sim_mat, topk, min_sim, dyn_q, margin_min, no_fallback, topk_max):
        h_size, w_size = sim_mat.shape
        row_topn = min(2, w_size)
        col_topn = min(2, h_size)
        row_vals, row_idx = torch.topk(sim_mat, k=row_topn, dim=1)
        col_vals, col_idx = torch.topk(sim_mat, k=col_topn, dim=0)

        candidate_rows = []
        candidate_rights = []
        candidate_sims = []
        candidate_margins = []
        candidate_quality = []
        for row in range(h_size):
            col = int(row_idx[row, 0].item())
            if int(col_idx[0, col].item()) != row:
                continue
            best_sim = float(row_vals[row, 0].item())
            row_second = float(row_vals[row, 1].item()) if row_topn > 1 else -1.0
            col_second = float(col_vals[1, col].item()) if col_topn > 1 else -1.0
            margin = min(best_sim - row_second, best_sim - col_second)
            candidate_rows.append(row)
            candidate_rights.append(col)
            candidate_sims.append(best_sim)
            candidate_margins.append(margin)
            candidate_quality.append(best_sim + margin)

        candidate_count = len(candidate_rows)
        dyn_thr = -1.0
        if dyn_q > 0.0 and candidate_quality:
            dyn_thr = float(np.quantile(np.array(candidate_quality, dtype=np.float32), dyn_q))
        quality_thr = dyn_thr if dyn_q > 0.0 else -1.0

        filtered = []
        for row, col, sim, margin, quality in zip(
            candidate_rows,
            candidate_rights,
            candidate_sims,
            candidate_margins,
            candidate_quality,
        ):
            if sim < min_sim:
                continue
            if margin < margin_min:
                continue
            if quality < quality_thr:
                continue
            filtered.append((quality, sim, margin, row, col))

        filtered.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        selected_cap = topk
        if topk_max > 0:
            selected_cap = min(selected_cap, topk_max)
        selected_target = topk
        if topk_max > 0:
            selected_target = min(selected_target, topk_max)
        visual_links = [
            (left_ents[row], right_ents[col]) for _, _, _, row, col in filtered[:selected_cap]
        ]
        fallback_added = 0
        if (not no_fallback) and len(visual_links) < selected_target:
            legacy = build_legacy_visual_links(sim_mat, selected_target, min_sim, dyn_q)
            existing = set(visual_links)
            for link in legacy["visual_links"]:
                if link in existing:
                    continue
                visual_links.append(link)
                existing.add(link)
                fallback_added += 1
                if len(visual_links) == selected_target:
                    break

        final_thr = max(min_sim, quality_thr if dyn_q > 0.0 else -1.0)
        return {
            "visual_links": visual_links,
            "candidate_count": candidate_count,
            "dyn_thr": dyn_thr,
            "final_thr": final_thr,
            "passed_threshold": len(filtered),
            "fallback_added": fallback_added,
            "filter_mode": "mutual_nearest_margin",
            "quality_thr": quality_thr,
            "margin_min": margin_min,
            "mutual_candidates": candidate_count,
        }

    l_img_f = img_features[left_ents]
    r_img_f = img_features[right_ents]
    img_sim = l_img_f.mm(r_img_f.t())

    topk = int(getattr(args, "unsup_k", 1000))
    topk_max = int(getattr(args, "unsup_k_max", 0))
    min_sim = float(getattr(args, "unsup_min_sim", -1.0))
    dyn_q = float(getattr(args, "unsup_dynamic_quantile", 0.0))
    dyn_q = min(1.0, max(0.0, dyn_q))
    use_bipartite_filter = bool(int(getattr(args, "unsup_use_bipartite_filter", 0)))
    margin_min = float(getattr(args, "unsup_margin_min", 0.0))
    no_fallback = bool(int(getattr(args, "unsup_no_fallback", 0)))

    if use_bipartite_filter:
        result = build_bipartite_visual_links(
            img_sim,
            topk=topk,
            min_sim=min_sim,
            dyn_q=dyn_q,
            margin_min=margin_min,
            no_fallback=no_fallback,
            topk_max=topk_max,
        )
    else:
        result = build_legacy_visual_links(
            img_sim,
            topk=topk,
            min_sim=min_sim,
            dyn_q=dyn_q,
        )

    del l_img_f, r_img_f, img_sim

    visual_links = result["visual_links"]
    count = 0.0
    for link in visual_links:
        if link in ills:
            count = count + 1
    ratio = 0.0 if len(visual_links) == 0 else (count / len(visual_links) * 100)
    logger.info(
        f"visual seed filter: topk={topk}, selected_cap={topk_max if topk_max > 0 else topk}, "
        f"candidate={result['candidate_count']}, mode={result['filter_mode']}, "
        f"unsup_min_sim={min_sim:.4f}, unsup_dynamic_q={dyn_q:.2f}, "
        f"dyn_thr={result['dyn_thr']:.4f}, final_thr={result['final_thr']:.4f}, "
        f"passed_thr={result['passed_threshold']}, fallback_added={result['fallback_added']}, "
        f"selected={len(visual_links)}"
    )
    if use_bipartite_filter:
        logger.info(
            f"visual seed mutual stats: mutual_candidates={result.get('mutual_candidates', 0)}, "
            f"margin_min={margin_min:.4f}, quality_thr={result.get('quality_thr', -1.0):.4f}, "
            f"no_fallback={int(no_fallback)}"
        )
    logger.info(f"{ratio:.2f}% in true links")
    logger.info(f"visual links length: {(len(visual_links))}")
    if len(visual_links) == 0:
        train_ill = np.empty((0, 2), dtype=np.int32)
    else:
        train_ill = np.array(visual_links, dtype=np.int32)
    return train_ill


def read_raw_data(file_dir, lang=[1, 2]):
    """
    Read DBP15k/DWY15k dataset.
    Parameters
    ----------
    file_dir: root of the dataset.
    Returns
    -------
    ent2id_dict : A dict mapping from entity name to ids
    ills: inter-lingual links (specified by ids)
    triples: a list of tuples (ent_id_1, relation_id, ent_id_2)
    r_hs: a dictionary containing mappings of relations to a list of entities that are head entities of the relation
    r_ts: a dictionary containing mappings of relations to a list of entities that are tail entities of the relation
    ids: all ids as a list
    """
    print('loading raw data...')

    def read_file(file_paths):
        tups = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    tups.append(tuple([int(x) for x in params]))
        return tups

    def read_dict(file_paths):
        ent2id_dict = {}
        ids = []
        for file_path in file_paths:
            id = set()
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id_dict, ids
    ent2id_dict, ids = read_dict([file_dir + "/ent_ids_" + str(i) for i in lang])
    ills = read_file([file_dir + "/ill_ent_ids"])
    triples = read_file([file_dir + "/triples_" + str(i) for i in lang])
    r_hs, r_ts = {}, {}
    for (h, r, t) in triples:
        if r not in r_hs:
            r_hs[r] = set()
        if r not in r_ts:
            r_ts[r] = set()
        r_hs[r].add(h)
        r_ts[r].add(t)
    assert len(r_hs) == len(r_ts)
    return ent2id_dict, ills, triples, r_hs, r_ts, ids


def loadfile(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ids(fn):
    ids = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ids.append(int(th[0]))
    return ids


def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id


# The most frequent attributes are selected to save space
def load_attr(fns, e, ent2id, topA=1000):
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    attr2id = {}
    # pdb.set_trace()
    topA = min(1000, len(fre))
    for i in range(topA):
        attr2id[fre[i][0]] = i
    attr = np.zeros((e, topA), dtype=np.float32)
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
    return attr


def load_relation(e, KG, topR=1000):
    # (39654, 1000)
    rel_mat = np.zeros((e, topR), dtype=np.float32)
    rels = np.array(KG)[:, 1]
    top_rels = Counter(rels).most_common(topR)
    rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}
    for tri in KG:
        h = tri[0]
        r = tri[1]
        o = tri[2]
        if r in rel_index_dict:
            rel_mat[h][rel_index_dict[r]] += 1.
            rel_mat[o][rel_index_dict[r]] += 1.
    return np.array(rel_mat)


def load_json_embd(path):
    embd_dict = {}
    with open(path) as f:
        for line in f:
            example = json.loads(line.strip())
            vec = np.array([float(e) for e in example['feature'].split()])
            embd_dict[int(example['guid'])] = vec
    return embd_dict


def load_img(logger, e_num, path, args=None):
    img_dict = pickle.load(open(path, "rb"))
    # init unknown img vector with mean and std deviation of the known's
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)
    # img_embd = np.array([np.zeros_like(img_dict[0]) for i in range(e_num)]) # no image
    # img_embd = np.array([img_dict[i] if i in img_dict else np.zeros_like(img_dict[0]) for i in range(e_num)])

    drop_rate = 0.0
    drop_seed = -1
    if args is not None:
        drop_rate = float(getattr(args, "img_mask_drop_rate", 0.0))
        drop_seed = int(getattr(args, "img_mask_drop_seed", -1))
    if drop_rate < 0.0 or drop_rate >= 1.0:
        raise ValueError(f"img_mask_drop_rate must be in [0, 1), got {drop_rate}")

    drop_ids = set()
    available_ids = [int(i) for i in img_dict.keys() if 0 <= int(i) < e_num]
    if drop_rate > 0.0 and len(available_ids) > 0:
        effective_seed = drop_seed if drop_seed >= 0 else int(getattr(args, "random_seed", 0))
        drop_count = int(round(len(available_ids) * drop_rate))
        if drop_count > 0:
            rng = np.random.default_rng(effective_seed)
            sampled = rng.choice(np.array(available_ids, dtype=np.int32), size=drop_count, replace=False)
            drop_ids = set(int(x) for x in sampled.tolist())
        logger.info(
            f"image pressure drop: requested_rate={drop_rate:.2f}, "
            f"drop_seed={effective_seed}, dropped={len(drop_ids)}/{len(available_ids)}"
        )

    img_mask = np.zeros((e_num,), dtype=np.float32)
    img_embd = []
    for i in range(e_num):
        if i in img_dict and i not in drop_ids:
            img_embd.append(img_dict[i])
            img_mask[i] = 1.0
        else:
            img_embd.append(np.random.normal(mean, std, mean.shape[0]))
    img_embd = np.array(img_embd)
    effective_have_img = len(available_ids) - len(drop_ids)
    logger.info(
        f"{(100 * effective_have_img / e_num):.2f}% entities have images"
        + (f" after pressure drop ({len(drop_ids)} removed)" if len(drop_ids) > 0 else "")
    )
    return img_embd, img_mask
