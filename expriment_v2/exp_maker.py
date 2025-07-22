import os
import json
import random
import argparse
from typing import List, Set, Dict, Tuple

FEATURES = [
    "original",
    "level1",
    "level2_1",
    "level2_2",
    "level2_3",
    "level3_sim0.5",
    "level3_sim0.7",
    "level3_sim0.9",
    "level3_0.8RE",
]

# 相对于 /home/yunxiang/work_june/source 的 6 条子路径（成员）
MEMBER_DIRS_REL = [
    "starcoder2-3b/analysis/memall",
    "starcoder2-3b/analysis/memall_nb",
    "starcoder2-3b/analysis_rec_new/memall",

    "starcoder2-7b/analysis/memall",
    "starcoder2-7b/analysis/memall_nb",
    "starcoder2-7b/analysis_rec_new/memall",

    "deepseekcoder-1.3b/analysis/memall",
    "deepseekcoder-1.3b/analysis/memall_nb",
    "deepseekcoder-1.3b/analysis_rec_new/memall",

    "deepseekcoder-6.7b/analysis/memall",
    "deepseekcoder-6.7b/analysis/memall_nb",
    "deepseekcoder-6.7b/analysis_rec_new/memall",

    "santacoder-1.1b/analysis/memall",
    "santacoder-1.1b/analysis/memall_nb",
    "santacoder-1.1b/analysis_rec_new/memall",

    "codellama-7b/analysis/memall",
    "codellama-7b/analysis/memall_nb",
    "codellama-7b/analysis_rec_new/memall",

    "deepseekcoder-33b/analysis/memall",
    "deepseekcoder-33b/analysis/memall_nb",
    "deepseekcoder-33b/analysis_rec_new/memall",

    "codellama-13b/analysis/memall",
    "codellama-13b/analysis/memall_nb",
    "codellama-13b/analysis_rec_new/memall"
]

# 非成员目录，通过简单替换得到
NONMEMBER_DIRS_REL = [p.replace("memall", "nmeall") for p in MEMBER_DIRS_REL]

SOURCE_BASE = "/home/yunxiang/work_june/source"

# 引入 level 编码映射，便于生成 per-level index 文件
from sampling_strategies import LEVEL_ORDER

def load_candidate_indices(feature_file: str, lb: int, rb: int) -> List[int]:
    """读取 *level3_0.8RE.jsonl_* 等特征文件，优先遍历其中的 index，
    然后在对应的 original.jsonl_ 中检查 input 字段长度是否符合要求，
    最终返回满足条件的 index 列表。

    参数说明：
        feature_file  : 形如 ``mem_level3_0.8RE.jsonl_`` 或 ``nme_level3_0.8RE.jsonl_`` 的文件路径
        lb, rb        : 输入(input) 字段长度的下界和上界（闭区间）
    """

    # 1. 收集候选 index —— 直接遍历 feature_file
    idx_set: Set[int] = set()
    with open(feature_file, "r", encoding="utf-8") as fr:
        for line in fr:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "index" in obj:
                idx_set.add(obj["index"])

    if not idx_set:
        return []

    # 2. 推断同目录下的 original.jsonl_ 文件路径
    dir_name, base_name = os.path.split(feature_file)
    # 将 feature 片段替换为 original
    # 例如 mem_level3_0.8RE.jsonl_ -> original.jsonl_
    orig_base = "original.jsonl_"
    original_file = os.path.join(dir_name, orig_base)

    if not os.path.exists(original_file):
        raise FileNotFoundError(f"推断的 original 文件不存在: {original_file}")

    # 3. 遍历 original 文件，对 idx_set 进行筛选
    accepted: List[int] = []
    remaining = set(idx_set)
    with open(original_file, "r", encoding="utf-8") as fr:
        for line in fr:
            if not remaining:
                break  # 提前结束
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = obj.get("index")
            if idx not in remaining:
                continue

            inp = obj.get("input", "")
            if inp is None:
                remaining.remove(idx)
                continue
            # 输入有可能是 list，统一转为 str 计算长度
            if not isinstance(inp, str):
                inp = str(inp)

            if lb <= len(inp) <= rb:
                accepted.append(idx)
            # 无论是否满足，都无需再检查该 idx
            remaining.remove(idx)

    return accepted


def sample_indices(candidates: List[int], size: int) -> Set[int]:
    if len(candidates) < size:
        raise ValueError("候选样本数量不足，无法采样指定数量的 index")
    return set(random.sample(candidates, size))


def get_feature_file_map(dir_abs: str) -> Dict[str, str]:
    """返回 {feature: filename} 的映射(仅文件名，不含路径)，确保每个特征恰有一个文件对应"""
    files = [f for f in os.listdir(dir_abs) if os.path.isfile(os.path.join(dir_abs, f))]
    mapping: Dict[str, str] = {}
    for feat in FEATURES:
        matched = [f for f in files if feat in f]
        if len(matched) != 1:
            raise FileNotFoundError(
                f"目录 {dir_abs} 下与特征 {feat} 对应的文件数目为 {len(matched)}，应当恰为 1"
            )
        mapping[feat] = matched[0]
    return mapping


def indices_exist_in_all_dirs(sample_set: Set[int], dirs_abs: List[str]) -> bool:
    """检查 sample_set 中的每个 index 是否在所有目录的 9 个特征文件中都存在"""
    for d in dirs_abs:
        try:
            feat_file_map = get_feature_file_map(d)
        except Exception as e:
            print(e)
            return False

        for feat, fname in feat_file_map.items():
            fpath = os.path.join(d, fname)
            found = set()
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        idx = json.loads(line)["index"]
                    except Exception:
                        continue
                    if idx in sample_set:
                        found.add(idx)
                        if len(found) == len(sample_set):
                            break
            if len(found) < len(sample_set):
                print(f"目录 {d} 下特征 {feat} 的文件 {fname} 缺失的 index: {sample_set - found}")
                # print(f"目录 {d} 下缺失的 index: {sample_set - found}")
                return False
    return True


def gather_lines_for_indices(sample_set: Set[int], dir_abs: str) -> Dict[str, List[str]]:
    """收集 dir_abs 中每个特征文件对应 sample_set 的行，返回 {basename: lines}"""
    feat_file_map = get_feature_file_map(dir_abs)
    res: Dict[str, List[str]] = {}
    for feat, fname in feat_file_map.items():
        fpath = os.path.join(dir_abs, fname)
        lines: List[str] = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("index") in sample_set:
                    lines.append(line.rstrip("\n"))
                if len(lines) == len(sample_set):
                    break
        if len(lines) != len(sample_set):
            raise RuntimeError(f"采样行收集失败: {fpath}")
        res[fname] = lines
    return res


def write_dataset(dest_root: str, dirs_rel: List[str], tag: int, sample_set: Set[int]):
    os.makedirs(dest_root, exist_ok=True)
    # 写 index-tag 文件
    index_path = os.path.join(dest_root, "index_tag.jsonl")
    with open(index_path, "w", encoding="utf-8") as fw:
        for idx in sorted(sample_set):
            fw.write(json.dumps({"index": idx, "tag": tag}, ensure_ascii=False) + "\n")

    for rel in dirs_rel:
        src_abs = os.path.join(SOURCE_BASE, rel)
        dest_abs = os.path.join(dest_root, rel)
        os.makedirs(dest_abs, exist_ok=True)
        lines_dict = gather_lines_for_indices(sample_set, src_abs)
        for basename, lines in lines_dict.items():
            dest_file = os.path.join(dest_abs, basename)
            with open(dest_file, "w", encoding="utf-8") as fw:
                for l in lines:
                    fw.write(l + "\n")

# ------------------------ 新增工具函数 ------------------------


def load_candidate_indices_len(feature_file: str, lb: int, rb: int) -> List[int]:
    """直接在 *feature_file* 中根据 input 长度筛选候选 index。

    仅检查自身文件，不再回退 original.jsonl_。
    """
    idxs: List[int] = []
    with open(feature_file, "r", encoding="utf-8") as fr:
        for line in fr:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            inp = obj.get("input", "")
            if not isinstance(inp, str):
                inp = str(inp)
            if lb <= len(inp) <= rb and "index" in obj:
                idxs.append(obj["index"])
    return idxs


def indices_exist_in_feature_all_dirs(sample_set: Set[int], dirs_abs: List[str], feature: str) -> bool:
    """检查 *sample_set* 中的每个 index 是否在 *dirs_abs* 的同名特征文件中均存在。"""
    for d in dirs_abs:
        try:
            feat_file_map = get_feature_file_map(d)
            fname = feat_file_map[feature]
        except Exception as e:
            print(e)
            return False

        fpath = os.path.join(d, fname)
        found = set()
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    idx = json.loads(line)["index"]
                except Exception:
                    continue
                if idx in sample_set:
                    found.add(idx)
                    if len(found) == len(sample_set):
                        break
        if len(found) < len(sample_set):
            missing = sample_set - found
            print(f"目录 {d} 的特征 {feature} 缺失 index: {missing}")
            return False
    return True


def write_dataset_per_level(dest_root: str, dirs_rel: List[str], tag: int, sample_sets: Dict[str, Set[int]]):
    """将 *sample_sets* 写入目标目录。

    sample_sets: {feature -> set(index)}
    对于 member (tag=1) / nonmember (tag=0) 各写一份 index_tag_<level>.jsonl，
    （自 2025-07 优化）不再复制原始样本行，避免巨量 I/O，只维护索引文件。
    """
    os.makedirs(dest_root, exist_ok=True)

    # 1) 写每级别独立的 index 列表文件
    for feat, idx_set in sample_sets.items():
        level_code = LEVEL_ORDER[feat]
        index_path = os.path.join(dest_root, f"index_tag_{level_code}.jsonl")
        with open(index_path, "w", encoding="utf-8") as fw:
            for idx in sorted(idx_set):
                fw.write(json.dumps({"index": idx, "tag": tag}, ensure_ascii=False) + "\n")

    # 2) 原始样本文件不再复制 —— 仅保留索引文件，实际数据在 source 目录读取。

# python3 exp_maker.py --exp_name exp1a --size 350 --lb 100 --rb 200 --seed 123
# python3 exp_maker.py --exp_name exp2a --size 350 --lb 200 --rb 300 --seed 123
# python3 exp_maker.py --exp_name exp3a --size 350 --lb 300 --rb 400 --seed 123
# python3 exp_maker.py --exp_name exp4a --size 350 --lb 400 --rb 500 --seed 123
# python3 exp_maker.py --exp_name exp5a --size 350 --lb 500 --rb 600 --seed 123
def main():
    parser = argparse.ArgumentParser(description="构造实验基础数据集 (baseset)")
    parser.add_argument("--exp_name", required=True, help="实验名称，例如 exp1")
    parser.add_argument("--size", type=int, required=True, help="采样样本数量")
    parser.add_argument("--lb", type=int, required=True, help="输入长度下界 (inclusive)")
    parser.add_argument("--rb", type=int, required=True, help="输入长度上界 (inclusive)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dest_base", default=os.path.join(os.path.dirname(__file__), "baseset"), help="目标根目录")

    args = parser.parse_args()
    random.seed(args.seed)

    experiment_root = os.path.join(args.dest_base, args.exp_name)

    # ------------------------ member ------------------------
    member_samples: Dict[str, Set[int]] = {}
    member_dirs_abs = [os.path.join(SOURCE_BASE, p) for p in MEMBER_DIRS_REL]

    for feat in FEATURES:
        if feat == "original":
            fname = "original.jsonl_"
        else:
            fname = f"mem_{feat}.jsonl_"
        feature_file = os.path.join(SOURCE_BASE, MEMBER_DIRS_REL[0], fname)
        candidates = load_candidate_indices_len(feature_file, args.lb, args.rb)
        if len(candidates) < args.size:
            raise RuntimeError(f"成员数据中 feature={feat} 候选不足 (found={len(candidates)} < size={args.size})")
        while True:
            sample_set = sample_indices(candidates, args.size)
            if indices_exist_in_feature_all_dirs(sample_set, member_dirs_abs, feat):
                member_samples[feat] = sample_set
                break
            print(f"[member] feature={feat} 采样集合存在缺失，重新采样……")

    write_dataset_per_level(os.path.join(experiment_root, "member"), MEMBER_DIRS_REL, tag=1, sample_sets=member_samples)

    # ------------------------ nonmember ------------------------
    nonmember_samples: Dict[str, Set[int]] = {}
    nonmember_dirs_abs = [os.path.join(SOURCE_BASE, p) for p in NONMEMBER_DIRS_REL]

    for feat in FEATURES:
        if feat == "original":
            fname = "original.jsonl_"
        else:
            fname = f"nme_{feat}.jsonl_"
        feature_file = os.path.join(SOURCE_BASE, NONMEMBER_DIRS_REL[0], fname)
        candidates = load_candidate_indices_len(feature_file, args.lb, args.rb)
        if len(candidates) < args.size:
            raise RuntimeError(f"非成员数据中 feature={feat} 候选不足 (found={len(candidates)} < size={args.size})")
        while True:
            sample_set = sample_indices(candidates, args.size)
            if indices_exist_in_feature_all_dirs(sample_set, nonmember_dirs_abs, feat):
                nonmember_samples[feat] = sample_set
                break
            print(f"[nonmember] feature={feat} 采样集合存在缺失，重新采样……")

    write_dataset_per_level(os.path.join(experiment_root, "nonmember"), NONMEMBER_DIRS_REL, tag=0, sample_sets=nonmember_samples)

    print(f"基础数据集已生成于: {experiment_root}")


if __name__ == "__main__":
    main()
