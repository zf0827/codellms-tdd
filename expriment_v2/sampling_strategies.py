import json
import random
from typing import List, Tuple

# ---------------------- 基础映射 ----------------------

# 特征到扰动级别的映射
LEVEL_ORDER = {
    "original": "0",
    "level1": "1",
    "level2_1": "2a",
    "level2_2": "2b",
    "level2_3": "2c",
    "level3_sim0.9": "3a",
    "level3_sim0.7": "3b",
    "level3_sim0.5": "3c",
    "level3_0.8RE": "3d"
}

# 反向映射：扰动级别到特征
LEVEL_TO_FEATURE = {v: k for k, v in LEVEL_ORDER.items()}

# ---------------------- 工具函数 ----------------------

def get_valid_level_sequence(type2: str) -> List[str]:
    """根据 type2 参数返回有效的扰动级别序列"""
    if type2 == "2a":
        return ["0", "1", "2a", "3a", "3b", "3c", "3d"]
    if type2 == "2b":
        return ["0", "1", "2b", "3a", "3b", "3c", "3d"]
    if type2 == "2c":
        return ["0", "1", "2c", "3a", "3b", "3c", "3d"]
    raise ValueError(f"不支持的 type2 值: {type2}")


def extract_feature_from_filename(basename: str) -> str:
    """从文件名中提取特征名（忽略前缀）。"""
    # 优先匹配最长的 key，避免 level3_sim0.9 被识别成 level3
    candidates = sorted(LEVEL_ORDER.keys(), key=lambda x: -len(x))
    for cand in candidates:
        if cand in basename:
            return cand
    if basename == "original.jsonl_":
        return "original"
    return basename.split('.jsonl_')[0]

# ---------------------- 采样向量 ----------------------

def create_sampling_vector(
    folder_0_files: List[str],
    folder_1_files: List[str],
    type2: str,
    sampling_strategy: callable,
    **kwargs
) -> Tuple[List[str], List[float], List[float]]:
    """根据采样策略创建 V0、V1 频率向量"""
    valid_levels = get_valid_level_sequence(type2)

    # 为两个"文件夹"分配扰动级别（这里的"文件"可以是占位符名称）
    folder_0_levels: List[str] = []
    folder_1_levels: List[str] = []
    for file in folder_0_files:
        feature = extract_feature_from_filename(file.split('/')[-1])
        if feature in LEVEL_ORDER:
            level = LEVEL_ORDER[feature]
            if level in valid_levels:
                folder_0_levels.append(level)
    for file in folder_1_files:
        feature = extract_feature_from_filename(file.split('/')[-1])
        if feature in LEVEL_ORDER:
            level = LEVEL_ORDER[feature]
            if level in valid_levels:
                folder_1_levels.append(level)

    # 按 valid_levels 顺序重新排序文件列表，确保 indices 对齐
    folder_0_sorted: List[str] = []
    folder_1_sorted: List[str] = []
    for lvl in valid_levels:
        for i, l in enumerate(folder_0_levels):
            if l == lvl:
                folder_0_sorted.append(folder_0_files[i])
        for i, l in enumerate(folder_1_levels):
            if l == lvl:
                folder_1_sorted.append(folder_1_files[i])

    all_files = folder_0_sorted + folder_1_sorted

    # 调用具体采样策略得到频率
    v0_props, v1_props = sampling_strategy(
        folder_0_files=folder_0_sorted,
        folder_1_files=folder_1_sorted,
        valid_levels=valid_levels,
        **kwargs
    )
    return all_files, v0_props, v1_props

# ---------------------- 采样策略实现 ----------------------
# 下面的每个策略函数都遵循相同接口，返回 (v0, v1) 两个频率向量。

def _init_vectors(total_len: int) -> Tuple[List[float], List[float]]:
    return [0.0] * total_len, [0.0] * total_len


def test1_basic_infringement(folder_0_files: List[str], folder_1_files: List[str],
                             valid_levels: List[str], K: int, **kwargs):
    """测试1：基础侵权检测 V0=S0, V1=S_{1≤K}"""
    v0, v1 = _init_vectors(len(folder_0_files) + len(folder_1_files))
    # V0: folder0 中 level=0
    for i, f in enumerate(folder_0_files):
        if LEVEL_ORDER[extract_feature_from_filename(f)] == "0":
            v0[i] = 1.0
            break
    # V1: folder1 中 level ≤ K
    for i, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) <= K:
            v1[i + len(folder_0_files)] = 1.0 / (K + 1)  # 均分
    return v0, v1


def test2_non_infringement(folder_0_files: List[str], folder_1_files: List[str],
                           valid_levels: List[str], K: int, **kwargs):
    """测试2：非侵权误判 V1=S1, V0=S_{1>K}"""
    v0, v1 = _init_vectors(len(folder_0_files) + len(folder_1_files))
    # V1: folder1 level=0
    for i, f in enumerate(folder_1_files):
        if LEVEL_ORDER[extract_feature_from_filename(f)] == "0":
            v1[i + len(folder_0_files)] = 1.0
            break
    # V0: folder1 level>K
    candidates = []
    for i, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) > K:
            candidates.append(i + len(folder_0_files))
    for idx in candidates:
        v0[idx] = 1.0 / len(candidates) if candidates else 0.0
    return v0, v1


def test3_perturbation_distinction(folder_0_files: List[str], folder_1_files: List[str],
                                   valid_levels: List[str], K: int, **kwargs):
    """测试3：扰动区分 V1=S_{1≤K}, V0=S_{1>K}"""
    v0, v1 = _init_vectors(len(folder_0_files) + len(folder_1_files))
    for i, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        idx_global = i + len(folder_0_files)
        if valid_levels.index(lvl) <= K:
            v1[idx_global] = 1.0 / (K + 1)
        else:
            v0[idx_global] = 1.0 / (len(valid_levels) - K - 1)
    return v0, v1


def test4_synchronized_perturbation(folder_0_files: List[str], folder_1_files: List[str],
                                    valid_levels: List[str], K: int, **kwargs):
    """测试4：同步扰动 V1=S_{1≤K}, V0=S_{0≤K}"""
    v0, v1 = _init_vectors(len(folder_0_files) + len(folder_1_files))
    # folder1 -> V1
    cnt_v1 = 0
    for i, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) <= K:
            v1[i + len(folder_0_files)] = 1.0  # 暂存
            cnt_v1 += 1
    if cnt_v1:
        for i in range(len(v1)):
            v1[i] /= cnt_v1
    # folder0 -> V0
    cnt_v0 = 0
    for i, f in enumerate(folder_0_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) <= K:
            v0[i] = 1.0
            cnt_v0 += 1
    if cnt_v0:
        for i in range(len(v0)):
            v0[i] /= cnt_v0
    return v0, v1


def test5_real_scenario(folder_0_files: List[str], folder_1_files: List[str],
                         valid_levels: List[str], K: int, **kwargs):
    """测试5：真实场景 V1=平均 S_{1≤K}, V0=0.7*S_{0i}+0.3*S_{1>K}"""
    v0, v1 = _init_vectors(len(folder_0_files) + len(folder_1_files))
    # V1
    candidates_v1 = []
    for i, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) <= K:
            candidates_v1.append(i + len(folder_0_files))
    for idx in candidates_v1:
        v1[idx] = 1.0 / len(candidates_v1) if candidates_v1 else 0.0
    # V0 part1: folder0 全部
    for i in range(len(folder_0_files)):
        v0[i] = 0.7 / len(folder_0_files)
    # V0 part2: folder1 level>K
    candidates_v0_2 = []
    for i, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) > K:
            candidates_v0_2.append(i + len(folder_0_files))
    for idx in candidates_v0_2:
        v0[idx] = 0.3 / len(candidates_v0_2) if candidates_v0_2 else 0.0
    return v0, v1


def test6_per_level_sync(folder_0_files: List[str], folder_1_files: List[str],
                          valid_levels: List[str], i: int, **kwargs):
    """测试6：逐级同步 V1=S_{1i}, V0=S_{0i}"""
    v0, v1 = _init_vectors(len(folder_0_files) + len(folder_1_files))
    target_level = valid_levels[i]
    target_feature = LEVEL_TO_FEATURE[target_level]
    for idx, f in enumerate(folder_0_files):
        if extract_feature_from_filename(f) == target_feature:
            v0[idx] = 1.0
            break
    for idx, f in enumerate(folder_1_files):
        if extract_feature_from_filename(f) == target_feature:
            v1[idx + len(folder_0_files)] = 1.0
            break
    return v0, v1


def test7_adjacent_levels(folder_0_files: List[str], folder_1_files: List[str],
                           valid_levels: List[str], i: int, **kwargs):
    """测试7：相邻级别 V1=S_{1i}, V0=S_{1(i+1)}"""
    v0, v1 = _init_vectors(len(folder_0_files) + len(folder_1_files))
    lvl_i = valid_levels[i]
    lvl_i1 = valid_levels[i + 1]
    feat_i = LEVEL_TO_FEATURE[lvl_i]
    feat_i1 = LEVEL_TO_FEATURE[lvl_i1]
    for idx, f in enumerate(folder_1_files):
        if extract_feature_from_filename(f) == feat_i:
            v1[idx + len(folder_0_files)] = 1.0
        if extract_feature_from_filename(f) == feat_i1:
            v0[idx + len(folder_0_files)] = 1.0
    return v0, v1


def test5_real_scenario_5050(folder_0_files: List[str], folder_1_files: List[str],
                              valid_levels: List[str], K: int, **kwargs):
    """测试5扩展：真实场景 (0.5, 0.5) 组合
    V1 = 平均 S_{1≤K}
    V0 = 0.5·tag0(全部等级) + 0.5·tag1(level>K)
    """
    v0, v1 = _init_vectors(len(folder_0_files) + len(folder_1_files))
    # V1 与原版保持一致
    candidates_v1: List[int] = []
    for i, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) <= K:
            candidates_v1.append(i + len(folder_0_files))
    for idx in candidates_v1:
        v1[idx] = 1.0 / len(candidates_v1) if candidates_v1 else 0.0

    # V0 part1: folder0 全部 (权重 0.5)
    if folder_0_files:
        for i in range(len(folder_0_files)):
            v0[i] = 0.5 / len(folder_0_files)

    # V0 part2: folder1 level> K (权重 0.5)
    candidates_v0_2: List[int] = []
    for i, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) > K:
            candidates_v0_2.append(i + len(folder_0_files))
    if candidates_v0_2:
        for idx in candidates_v0_2:
            v0[idx] = 0.5 / len(candidates_v0_2)
    return v0, v1


def test5_real_scenario_3070(folder_0_files: List[str], folder_1_files: List[str],
                              valid_levels: List[str], K: int, **kwargs):
    """测试5扩展：真实场景 (0.3, 0.7) 组合
    V1 = 平均 S_{1≤K}
    V0 = 0.3·tag0(全部等级) + 0.7·tag1(level>K)
    """
    v0, v1 = _init_vectors(len(folder_0_files) + len(folder_1_files))
    # V1 与原版保持一致
    candidates_v1: List[int] = []
    for i, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) <= K:
            candidates_v1.append(i + len(folder_0_files))
    for idx in candidates_v1:
        v1[idx] = 1.0 / len(candidates_v1) if candidates_v1 else 0.0

    # V0 part1: folder0 全部 (权重 0.3)
    if folder_0_files:
        for i in range(len(folder_0_files)):
            v0[i] = 0.3 / len(folder_0_files)

    # V0 part2: folder1 level> K (权重 0.7)
    candidates_v0_2: List[int] = []
    for i, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) > K:
            candidates_v0_2.append(i + len(folder_0_files))
    if candidates_v0_2:
        for idx in candidates_v0_2:
            v0[idx] = 0.7 / len(candidates_v0_2)
    return v0, v1


def test8_realistic_scenario(folder_0_files: List[str], folder_1_files: List[str],
                              valid_levels: List[str], K: int, **kwargs):
    """测试8：增强真实场景
    V1：100% * tag1 level∈[0, K] (各级平均)
    V0：80% 来自 tag0，内部构成为
            • 25% * level0
            • 15% * level1
            • 15% * level2 (2a/2b/2c)
            • 15% * level3a–3c (平均)
            • 10% * level3d
         20% 来自 tag1 level>K (平均)
    参数说明：K 同前，代表扰动等级上界（含）。
    """
    total_len = len(folder_0_files) + len(folder_1_files)
    v0, v1 = _init_vectors(total_len)

    # -------------------- V1 --------------------
    # 100% 权重在 tag1 level∈[0, K]
    indices_v1_up_to_k: List[int] = []
    for idx, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        #valid_levels.index(lvl) <= K
        if valid_levels.index(lvl) <= K:
            indices_v1_up_to_k.append(idx + len(folder_0_files))
    if indices_v1_up_to_k:
        w = 1.0 / len(indices_v1_up_to_k)
        for i in indices_v1_up_to_k:
            v1[i] = w

    # -------------------- V0 --------------------
    def _assign(indices: List[int], weight: float):
        if indices:
            w_each = weight / len(indices)
            for i in indices:
                v0[i] = w_each

    # tag0 部分（80%）
    # a) level0 25% of 80% -> 0.20
    lvl0_indices = [i for i, f in enumerate(folder_0_files)
                    if LEVEL_ORDER[extract_feature_from_filename(f)] == "0"]
    _assign(lvl0_indices, 0.20)

    # b) level1 15% of 80% -> 0.12
    lvl1_indices = [i for i, f in enumerate(folder_0_files)
                    if LEVEL_ORDER[extract_feature_from_filename(f)] == "1"]
    _assign(lvl1_indices, 0.12)

    # c) level2 (2a/2b/2c) 15% of 80% -> 0.12
    lvl2_indices = [i for i, f in enumerate(folder_0_files)
                    if LEVEL_ORDER[extract_feature_from_filename(f)] in {"2a", "2b", "2c"}]
    _assign(lvl2_indices, 0.12)

    # d) level3a–3c 15% of 80% -> 0.12（平均分布）
    lvl3abc_indices = [i for i, f in enumerate(folder_0_files)
                       if LEVEL_ORDER[extract_feature_from_filename(f)] in {"3a", "3b", "3c"}]
    _assign(lvl3abc_indices, 0.12)

    # e) level3d 10% of 80% -> 0.08
    lvl3d_indices = [i for i, f in enumerate(folder_0_files)
                     if LEVEL_ORDER[extract_feature_from_filename(f)] == "3d"]
    _assign(lvl3d_indices, 0.08)

    # tag1 level>K 部分（20%）
    indices_v0_tag1_high: List[int] = []
    for idx, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) > K:
            indices_v0_tag1_high.append(idx + len(folder_0_files))
    _assign(indices_v0_tag1_high, 0.20)

    return v0, v1


def test8_limit(folder_0_files: List[str], folder_1_files: List[str],
                valid_levels: List[str], K: int, **kwargs):
    """测试8限制版：更贴近真实情景
    V1：100% * tag1 level∈[0, K] (各级、各样本平均)
    V0：
        80% * tag0 level∈[0, K] (平均)
        20% * tag1 level>K (平均)
    若 K 达到最大等级(6)，则应在 attempt_maker 中跳过此策略。
    """
    total_len = len(folder_0_files) + len(folder_1_files)
    v0, v1 = _init_vectors(total_len)

    # -------------------- V1 --------------------
    indices_v1: List[int] = []
    for idx, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) <= K:
            indices_v1.append(idx + len(folder_0_files))
    if indices_v1:
        w_each = 1.0 / len(indices_v1)
        for i in indices_v1:
            v1[i] = w_each

    # -------------------- V0 --------------------
    # tag0 部分 80%
    indices_v0_tag0: List[int] = [i for i, f in enumerate(folder_0_files)
                                  if valid_levels.index(LEVEL_ORDER[extract_feature_from_filename(f)]) <= K]
    if indices_v0_tag0:
        w_tag0 = 0.80 / len(indices_v0_tag0)
        for i in indices_v0_tag0:
            v0[i] = w_tag0

    # tag1 部分 20% (>K)
    indices_v0_tag1_high: List[int] = []
    for idx, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) > K:
            indices_v0_tag1_high.append(idx + len(folder_0_files))
    if indices_v0_tag1_high:
        w_tag1 = 0.20 / len(indices_v0_tag1_high)
        for i in indices_v0_tag1_high:
            v0[i] = w_tag1

    return v0, v1

def test8_limit_2080(folder_0_files: List[str], folder_1_files: List[str],
                valid_levels: List[str], K: int, **kwargs):
    """测试8限制版：更贴近真实情景
    V1：100% * tag1 level∈[0, K] (各级、各样本平均)
    V0：
        20% * tag0 level∈[0, K] (平均)
        80% * tag1 level>K (平均)
    若 K 达到最大等级(6)，则应在 attempt_maker 中跳过此策略。
    """
    total_len = len(folder_0_files) + len(folder_1_files)
    v0, v1 = _init_vectors(total_len)

    # -------------------- V1 --------------------
    indices_v1: List[int] = []
    for idx, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) <= K:
            indices_v1.append(idx + len(folder_0_files))
    if indices_v1:
        w_each = 1.0 / len(indices_v1)
        for i in indices_v1:
            v1[i] = w_each

    # -------------------- V0 --------------------
    # tag0 部分 80%
    indices_v0_tag0: List[int] = [i for i, f in enumerate(folder_0_files)
                                  if valid_levels.index(LEVEL_ORDER[extract_feature_from_filename(f)]) <= K]
    if indices_v0_tag0:
        w_tag0 = 0.20 / len(indices_v0_tag0)
        for i in indices_v0_tag0:
            v0[i] = w_tag0

    # tag1 部分 20% (>K)
    indices_v0_tag1_high: List[int] = []
    for idx, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) > K:
            indices_v0_tag1_high.append(idx + len(folder_0_files))
    if indices_v0_tag1_high:
        w_tag1 = 0.80 / len(indices_v0_tag1_high)
        for i in indices_v0_tag1_high:
            v0[i] = w_tag1

    return v0, v1

def test8_limit_5050(folder_0_files: List[str], folder_1_files: List[str],
                valid_levels: List[str], K: int, **kwargs):
    """测试8限制版：更贴近真实情景
    V1：100% * tag1 level∈[0, K] (各级、各样本平均)
    V0：
        50% * tag0 level∈[0, K] (平均)
        50% * tag1 level>K (平均)
    若 K 达到最大等级(6)，则应在 attempt_maker 中跳过此策略。
    """
    total_len = len(folder_0_files) + len(folder_1_files)
    v0, v1 = _init_vectors(total_len)

    # -------------------- V1 --------------------
    indices_v1: List[int] = []
    for idx, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) <= K:
            indices_v1.append(idx + len(folder_0_files))
    if indices_v1:
        w_each = 1.0 / len(indices_v1)
        for i in indices_v1:
            v1[i] = w_each

    # -------------------- V0 --------------------
    # tag0 部分 80%
    indices_v0_tag0: List[int] = [i for i, f in enumerate(folder_0_files)
                                  if valid_levels.index(LEVEL_ORDER[extract_feature_from_filename(f)]) <= K]
    if indices_v0_tag0:
        w_tag0 = 0.50 / len(indices_v0_tag0)
        for i in indices_v0_tag0:
            v0[i] = w_tag0

    # tag1 部分 20% (>K)
    indices_v0_tag1_high: List[int] = []
    for idx, f in enumerate(folder_1_files):
        lvl = LEVEL_ORDER[extract_feature_from_filename(f)]
        if valid_levels.index(lvl) > K:
            indices_v0_tag1_high.append(idx + len(folder_0_files))
    if indices_v0_tag1_high:
        w_tag1 = 0.50 / len(indices_v0_tag1_high)
        for i in indices_v0_tag1_high:
            v0[i] = w_tag1

    return v0, v1
# ---------------------- 策略映射 ----------------------
SAMPLING_STRATEGIES = {
    "test1_basic_infringement": test1_basic_infringement,
    "test2_non_infringement": test2_non_infringement,
    "test3_perturbation_distinction": test3_perturbation_distinction,
    "test4_synchronized_perturbation": test4_synchronized_perturbation,
    "test5_real_scenario": test5_real_scenario,
    "test5_real_scenario_5050": test5_real_scenario_5050,
    "test5_real_scenario_3070": test5_real_scenario_3070,
    "test6_per_level_sync": test6_per_level_sync,
    "test7_adjacent_levels": test7_adjacent_levels,
    "test8_realistic_scenario": test8_realistic_scenario,
    "test8_limit": test8_limit,
    "test8_limit_2080": test8_limit_2080,
    "test8_limit_5050": test8_limit_5050,
}
