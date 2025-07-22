import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

# 引入新的采样策略文件
from sampling_strategies import (
    SAMPLING_STRATEGIES,
    create_sampling_vector,
    get_valid_level_sequence,
    LEVEL_ORDER,
    LEVEL_TO_FEATURE,
)

# ----------------- 通用工具函数 -----------------

def load_index_list(index_file: Path) -> List[int]:
    """加载 index_tag.jsonl，返回 index 列表（保持顺序）。"""
    index_list: List[int] = []
    with open(index_file, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                obj = json.loads(line)
                index_list.append(obj["index"])
            except Exception:
                continue
    return index_list


def build_placeholder_files(prefix: str, valid_levels: List[str]) -> List[str]:
    """基于扰动级别生成占位文件名列表，便于沿用 create_sampling_vector 逻辑。"""
    placeholders: List[str] = []
    for lvl in valid_levels:
        feature = LEVEL_TO_FEATURE[lvl]
        placeholders.append(f"{prefix}_{feature}.jsonl_")
    return placeholders


def normalize_and_allocate(proportions: List[float], total: int) -> List[int]:
    """把比例向量转成非负整数计数, 确保总和 == total。"""
    n = len(proportions)
    if n == 0:
        return []

    # 若所有比例为 0，则平均分配
    if sum(proportions) == 0:
        proportions = [1.0 / n] * n
    else:
        proportions = [p / sum(proportions) for p in proportions]

    # 初步向下取整
    float_counts = [p * total for p in proportions]
    raw_counts = [int(c) for c in float_counts]  # 向下取整

    diff = total - sum(raw_counts)  # 需要再补 diff 个样本

    # 根据余数大小分配剩余 diff
    remainders = [(float_counts[i] - raw_counts[i], i) for i in range(n)]
    remainders.sort(reverse=True)  # 余数大的优先

    for k in range(diff):
        idx = remainders[k % n][1]
        raw_counts[idx] += 1

    # 保险：确保不存在负值
    raw_counts = [max(0, c) for c in raw_counts]
    return raw_counts


def sample_indices(src_indices: List[int], k: int) -> List[int]:
    """从 src_indices 中随机采样 k 个 index。若 k 超过长度，则允许重复采样。"""
    # print(f"src_indices: {src_indices}, k: {k}")
    if k <= len(src_indices):
        return random.sample(src_indices, k)
    # 不足则放宽为有放回采样
    return random.choices(src_indices, k=k)

# ----------------- 基础加载函数 -----------------

def load_level_index_map(base_folder: Path) -> Dict[str, List[int]]:
    """读取 base_folder 下所有 index_tag_<level>.jsonl，返回 {level: indices}"""
    level_map: Dict[str, List[int]] = {}
    for lvl in LEVEL_TO_FEATURE.keys():  # 使用全部 9 个短码
        f = base_folder / f"index_tag_{lvl}.jsonl"
        if f.exists():
            level_map[lvl] = load_index_list(f)
        else:
            level_map[lvl] = []
    return level_map

# ----------------- 核心函数 -----------------

def create_dataset_with_sampling(
    indices_tag0_by_level: Dict[str, List[int]],
    indices_tag1_by_level: Dict[str, List[int]],
    out_file: Path,
    strategy_name: str,
    type2: str,
    sample_per_label: int,
    **kwargs,
):
    """使用采样策略构建 attempt 级别数据集（仅包含 index/tag/level/label）。"""
    if strategy_name not in SAMPLING_STRATEGIES:
        raise ValueError(f"未知采样策略: {strategy_name}")
    sampling_strategy = SAMPLING_STRATEGIES[strategy_name]

    valid_levels = get_valid_level_sequence(type2)
    # 为 folder0 (tag=0) 与 folder1 (tag=1) 构造占位列表
    folder0 = build_placeholder_files("tag0", valid_levels)
    folder1 = build_placeholder_files("tag1", valid_levels)

    # 获取采样向量
    all_files, v0_props, v1_props = create_sampling_vector(
        folder_0_files=folder0,
        folder_1_files=folder1,
        type2=type2,
        sampling_strategy=sampling_strategy,
        **kwargs,
    )

    # 预先准备输出目录
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 生成 label=0,1 的样本
    records: List[dict] = []

    total_levels = len(valid_levels)

    # ---------- 处理 label=0 ----------
    counts_label0 = normalize_and_allocate(v0_props, sample_per_label)
    for global_idx, cnt in enumerate(counts_label0):
        if cnt == 0:
            continue
        tag = 0 if global_idx < total_levels else 1
        level_idx = global_idx % total_levels
        level_code = valid_levels[level_idx]
        src_map = indices_tag0_by_level if tag == 0 else indices_tag1_by_level
        src_indices = src_map.get(level_code, [])
        if not src_indices:
            raise ValueError(f"tag={tag}, level={level_code} 无可用样本！")
        sampled = sample_indices(src_indices, cnt)
        for idx_val in sampled:
            records.append({
                "index": idx_val,
                "tag": tag,
                "level": level_code,
                "label": 0,
            })

    # ---------- 处理 label=1 ----------
    counts_label1 = normalize_and_allocate(v1_props, sample_per_label)
    for global_idx, cnt in enumerate(counts_label1):
        if cnt == 0:
            continue
        tag = 0 if global_idx < total_levels else 1
        level_idx = global_idx % total_levels
        level_code = valid_levels[level_idx]
        src_map = indices_tag0_by_level if tag == 0 else indices_tag1_by_level
        src_indices = src_map.get(level_code, [])
        if not src_indices:
            raise ValueError(f"tag={tag}, level={level_code} 无可用样本！")
        sampled = sample_indices(src_indices, cnt)
        for idx_val in sampled:
            records.append({
                "index": idx_val,
                "tag": tag,
                "level": level_code,
                "label": 1,
            })

    # 打散并保存
    random.shuffle(records)
    with open(out_file, "w", encoding="utf-8") as fout:
        for item in records:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    # print(f"已创建数据集: {out_file} (label0={sample_per_label}, label1={sample_per_label})")

# ----------------- 主入口 -----------------

# python3 attempt_maker.py --exp_name exp1a --attempt_id 1 --sample_per_label 350
def main():
    parser = argparse.ArgumentParser(
        description="根据基础数据集构建 attempt 数据集，仅生成 jsonl 文件"
    )
    parser.add_argument("--exp_name", required=True, help="实验名称，例如 exp1")
    parser.add_argument("--attempt_id", type=int, default=1, help="attempt 序号，默认 1")
    parser.add_argument(
        "--sample_per_label",
        type=int,
        default=1000,
        help="每个标签(0/1)在每个测试数据集中的样本数量",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="若目标 jsonl 已存在，则跳过生成 (增量式)"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    baseset_dir = base_dir / "baseset" / args.exp_name
    dataset_dir = base_dir / "dataset" / args.exp_name / f"attempt{args.attempt_id}"

    # 加载按 level 划分的 index
    indices_tag0_by_level = load_level_index_map(baseset_dir / "nonmember")
    indices_tag1_by_level = load_level_index_map(baseset_dir / "member")

    cnt_tag0_total = sum(len(v) for v in indices_tag0_by_level.values())
    cnt_tag1_total = sum(len(v) for v in indices_tag1_by_level.values())
    print(f"加载 index 完成: tag0_total={cnt_tag0_total}, tag1_total={cnt_tag1_total}")

    # 创建输出目录（若已存在则不强制覆盖）
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 测试 0：原始 MIA 数据集 ----------
    out_file_test0 = dataset_dir / "test0_originalMIA.jsonl"
    if out_file_test0.exists() and args.skip_existing:
        print(f"跳过生成 test0_originalMIA，文件已存在: {out_file_test0}")
    elif not out_file_test0.exists() or not args.skip_existing:
        records_test0: List[dict] = []
        # 仅使用 level=0 的索引集
        for idx in indices_tag0_by_level.get("0", []):
            records_test0.append({
                "index": idx,
                "tag": 0,
                "level": "0",
                "label": 0,
            })
        for idx in indices_tag1_by_level.get("0", []):
            records_test0.append({
                "index": idx,
                "tag": 1,
                "level": "0",
                "label": 1,
            })
        # 保存（随机打散）
        random.shuffle(records_test0)
        with open(out_file_test0, "w", encoding="utf-8") as fout:
            for item in records_test0:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"已创建数据集: {out_file_test0} (total={len(records_test0)})")

    # 定义测试组合 (与旧 main.py 保持一致)
    # 扰动等级共有 7 档(0,1,2a/2b/2c,3a,3b,3c,3d)，因此 K 的有效取值为 0–6。
    k_values = list(range(7))  # [0, 1, 2, 3, 4, 5, 6]
    type2_values = ["2a", "2b", "2c"]

    # ---------- 测试 1-5 ----------
    for K in k_values:
        for type2 in type2_values:
            subdir = dataset_dir / f"K{K}_{type2}"
            subdir.mkdir(parents=True, exist_ok=True)
            # 当 K 达到最大等级(6)时，测试2/3/5 中关于 "level>K" 的条件将不再成立，语义失效，故跳过。
            test_configs_all = [
                ("test1_basic_infringement", dict(K=K)),
                # ("test2_non_infringement", dict(K=K)),
                # ("test3_perturbation_distinction", dict(K=K)),
                # ("test4_synchronized_perturbation", dict(K=K)),
                # ("test5_real_scenario", dict(K=K)),
                # ("test5_real_scenario_5050", dict(K=K)),
                # ("test5_real_scenario_3070", dict(K=K)),
                # ("test8_realistic_scenario", dict(K=K)),
                ("test8_limit", dict(K=K)),
                ("test8_limit_2080", dict(K=K)),
                ("test8_limit_5050", dict(K=K)),
            ]
            if K == 6:
                # 过滤掉与 "level>K" 相关、在 K=6 时无意义的策略
                test_configs = [tc for tc in test_configs_all if tc[0] not in {
                    # "test2_non_infringement",
                    # "test3_perturbation_distinction",
                    # "test5_real_scenario",
                    # "test5_real_scenario_5050",
                    # "test5_real_scenario_3070",
                    # "test8_realistic_scenario",
                    "test8_limit",
                    "test8_limit_2080",
                    "test8_limit_5050",
                }]
            else:
                test_configs = test_configs_all
            for strategy_name, extra_params in test_configs:
                out_file = subdir / f"{strategy_name}.jsonl"
                if out_file.exists() and args.skip_existing:
                    # 增量模式：文件已存在，跳过
                    continue
                create_dataset_with_sampling(
                    indices_tag0_by_level,
                    indices_tag1_by_level,
                    out_file,
                    strategy_name,
                    type2,
                    sample_per_label=args.sample_per_label,
                    **extra_params,
                )

    # ---------- 测试 6 ----------
    for type2 in type2_values:
        valid_levels = get_valid_level_sequence(type2)
        for i in range(len(valid_levels)):
            subdir = dataset_dir / f"i{i}_{type2}"
            subdir.mkdir(parents=True, exist_ok=True)
            out_file = subdir / "test6_per_level_sync.jsonl"
            if out_file.exists() and args.skip_existing:
                continue
            create_dataset_with_sampling(
                indices_tag0_by_level,
                indices_tag1_by_level,
                out_file,
                "test6_per_level_sync",
                type2,
                sample_per_label=args.sample_per_label,
                i=i,
            )

    # ---------- 测试 7 ----------
    for type2 in type2_values:
        valid_levels = get_valid_level_sequence(type2)
        for i in range(len(valid_levels) - 1):
            subdir = dataset_dir / f"i{i}_vs_{i+1}_{type2}"
            subdir.mkdir(parents=True, exist_ok=True)
            out_file = subdir / "test7_adjacent_levels.jsonl"
            if out_file.exists() and args.skip_existing:
                continue
            create_dataset_with_sampling(
                indices_tag0_by_level,
                indices_tag1_by_level,
                out_file,
                "test7_adjacent_levels",
                type2,
                sample_per_label=args.sample_per_label,
                i=i,
            )

    print("所有 attempt 数据集已生成完毕！")


if __name__ == "__main__":
    main()
