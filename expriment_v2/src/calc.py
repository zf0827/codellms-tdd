import numpy as np
import zlib
import logging
import traceback

# 配置日志记录器
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def calculate_ppl(all_log_probs: list[float]) -> tuple[float, float | None]:
    """
    根据对数概率列表计算困惑度 (PPL)。

    Args:
        all_log_probs (list[float]): 对数概率列表。

    Returns:
        tuple[float, float | None]: (PPL 值, 平均对数概率)。
                                     如果无法计算，PPL 为 NaN，平均对数概率为 None 或 NaN。
    """
    if not all_log_probs:
        return float('nan'), None
    try:
        avg_log_prob = np.mean(all_log_probs)
        if avg_log_prob is not None and not np.isnan(avg_log_prob):
            ppl = np.exp(-avg_log_prob).item()
            return ppl, avg_log_prob
        else:
            return float('nan'), avg_log_prob
    except Exception as e:
        logger.error(f"计算 PPL 时发生错误: {e}")
        traceback.print_exc()
        return float('nan'), None


def calculate_zlib_entropy(text: str) -> int:
    """计算文本的 zlib 熵。"""
    if isinstance(text, (str, bytes)):
        try:
            return len(zlib.compress(bytes(str(text), 'utf-8')))
        except Exception as e:
            logger.error(f"计算 zlib 熵时出错: {e}")
            return 0
    return 0


def calculate_ppl_zlib(text: str, avg_log_prob: float | None, ppl_val: float) -> float:
    """计算 PPL/zlib。"""
    zlib_entropy = calculate_zlib_entropy(text)
    if zlib_entropy == 0:
        return float('nan')

    ppl_zlib_score = float('nan')
    if not np.isnan(ppl_val):
        try:
            if avg_log_prob is not None and not np.isnan(avg_log_prob):
                ppl_zlib_score = (-avg_log_prob) / zlib_entropy
            elif not np.isnan(ppl_val) and ppl_val > 0:
                ppl_zlib_score = np.log(ppl_val) / zlib_entropy
        except Exception as e:
            logger.error(f"计算 PPL/zlib 时发生错误: {e}")
            traceback.print_exc()
    return ppl_zlib_score


def calculate_mink_scores(all_prob: list[float]) -> dict:
    """计算 Min-k Prob 分数。"""
    mink_scores = {}
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4]

    if not all_prob:
        for ratio in ratios:
            mink_scores[f"Min_{int(ratio*100)}%"] = float('nan')
        return mink_scores

    try:
        for ratio in ratios:
            k_length = int(len(all_prob) * ratio)
            sorted_probs = np.sort(all_prob)
            topk_prob = sorted_probs[:k_length]

            if k_length == 0:
                mink_scores[f"Min_{int(ratio*100)}%"] = float('nan')
            else:
                mink_scores[f"Min_{int(ratio*100)}%"] = -np.mean(topk_prob).item()

        reverse_ratios = sorted(ratios, reverse=True)
        for i in range(len(reverse_ratios)):
            curr_key = f"Min_{int(reverse_ratios[i]*100)}%"
            if curr_key in mink_scores and np.isnan(mink_scores[curr_key]):
                for j in range(i - 1, -1, -1):
                    prev_key = f"Min_{int(reverse_ratios[j]*100)}%"
                    if prev_key in mink_scores and not np.isnan(mink_scores[prev_key]):
                        mink_scores[curr_key] = mink_scores[prev_key]
                        break
    except Exception as e:
        logger.error(f"计算 Min-K Prob 时发生错误: {e}")
        traceback.print_exc()
        for ratio in ratios:
            mink_scores[f"Min_{int(ratio*100)}%"] = float('nan')
    return mink_scores


def _get_loss_from_logprobs(lp_list: list[float]) -> float:
    """给定 logprob 序列，返回 Loss=-avg_logprob。若 lp_list 为空返回 NaN。"""
    if not lp_list:
        return float('nan')
    try:
        return -float(np.mean(lp_list))
    except Exception:
        return float('nan')


def _calculate_min_k_plus_scores(lp_list: list[float], ratios: list[float] | None = None) -> dict:
    """实现规范中的 Min_K%++ 指标。

    步骤：
        1. 归一化 logprob 列表，使用 z-score 标准化：
           s_i = (lp_i - μ) / σ
           其中 μ 是 logprob 的均值，σ 是标准差
        2. 取最小的 K% s_i，平均后取负数。

    返回字典键名形如 'Min_{ratio*100}%++'。
    """
    result: dict = {}
    if ratios is None:
        ratios = [0.05, 0.1, 0.2, 0.3, 0.4]

    if not lp_list:
        for r in ratios:
            result[f"Min_{int(r*100)}%++"] = float('nan')
        return result

    try:
        lp_arr = np.array(lp_list, dtype=float)
        # 计算均值和标准差
        mu = np.mean(lp_arr)
        sigma = np.std(lp_arr) + 1e-6  # 添加小常数避免除以零
        
        # 执行 z-score 标准化
        norm_scores = (lp_arr - mu) / sigma

        sorted_norm = np.sort(norm_scores)  # 从小到大
        n = len(sorted_norm)
        for r in ratios:
            k = max(int(n * r), 1)  # 至少取 1
            avg_val = float(np.mean(sorted_norm[:k]))
            result[f"Min_{int(r*100)}%++"] = -avg_val
    except Exception as e:
        logger.error(f"计算 Min_K%++ 时发生异常: {e}")
        traceback.print_exc()
        for r in ratios:
            result[f"Min_{int(r*100)}%++"] = float('nan')
    return result


def calculate_all_scores(sample: dict) -> dict:
    """根据样本字典计算分数 (多模型版本)。

    约定 sample 中包含字段：
        text,
        <model>_logprobs,
        <model>_nb_logprobs,
        <model>_rec_new_Loss
    其中 <model> 取自 run.BaseSetAccessor._MODEL_DIRS 的键。
    为每个模型分别计算以下指标并在键前加上模型前缀：
        ppl, ppl/zlib, Min_k, Min_k++, Ref, Neighbor, ReCall_new
    """
    text: str = sample.get("text", "") or ""
    # --------------------------- 解析模型列表 ---------------------------
    model_keys: list[str] = []
    for k in sample.keys():
        if k.endswith("_logprobs") and "nb" not in k:
            model_keys.append(k[:-len("_logprobs")])
    model_keys = sorted(set(model_keys))

    if not model_keys:
        logger.error("样本中未找到任何 *_logprobs 字段！")
        return {}

    scores: dict = {}
    zlib_entropy_val = calculate_zlib_entropy(text)
    scores["zlib_entropy"] = zlib_entropy_val  # 可选：与模型无关

    # 记录每个模型的 loss 以便稍后计算 Ref
    model_loss_map: dict[str, float] = {}

    # --------------------------- 主循环 ---------------------------
    for m in model_keys:
        lp = sample.get(f"{m}_logprobs", [])
        nb_lp = sample.get(f"{m}_nb_logprobs", [])
        rec_new_loss_raw = sample.get(f"{m}_rec_new_Loss", float('nan'))
        try:
            rec_new_loss_val = float(rec_new_loss_raw)
        except Exception:
            rec_new_loss_val = float('nan')

        # ppl & avg_lp
        ppl_val, avg_lp_val = calculate_ppl(lp)
        scores[f"{m}_ppl"] = ppl_val

        # ppl/zlib
        if zlib_entropy_val == 0:
            scores[f"{m}_ppl/zlib"] = float('nan')
        else:
            scores[f"{m}_ppl/zlib"] = calculate_ppl_zlib(text, avg_lp_val, ppl_val)

        # Min-K 及 Min-K++
        mink_scores = calculate_mink_scores(lp)
        scores.update({f"{m}_{k}": v for k, v in mink_scores.items()})

        minkpp_scores = _calculate_min_k_plus_scores(lp)
        scores.update({f"{m}_{k}": v for k, v in minkpp_scores.items()})

        # Neighbor
        _, nb_avg_lp = calculate_ppl(nb_lp)
        loss_orig = _get_loss_from_logprobs(lp)
        loss_nb = _get_loss_from_logprobs(nb_lp)
        scores[f"{m}_Neighbor"] = (
            loss_orig - loss_nb
            if not np.isnan(loss_orig) and not np.isnan(loss_nb) else float('nan')
        )

        # ReCall_new
        scores[f"{m}_ReCall_new"] = (
            -rec_new_loss_val / loss_orig
            if not np.isnan(rec_new_loss_val) and not np.isnan(loss_orig) and not np.isclose(loss_orig, 0)
            else float('nan')
        )

        # 记录 loss 供 Ref 计算
        model_loss_map[m] = loss_orig

        # 调试用原始 loss
        scores[f"{m}_loss"] = loss_orig
        scores[f"{m}_nb_loss"] = loss_nb
        scores[f"{m}_rec_new_Loss"] = rec_new_loss_val
    # --------------------------- Ref 指标 ---------------------------
    # 仅计算 starcoder2 与 deepseekcoder 两个 family 的 Ref
    if "starcoder2_7b" in model_loss_map and "starcoder2_3b" in model_loss_map:
        big_loss = model_loss_map["starcoder2_7b"]
        small_loss = model_loss_map["starcoder2_3b"]
        scores["starcoder2_7b_Ref"] = (
            big_loss - small_loss
            if not np.isnan(big_loss) and not np.isnan(small_loss) else float('nan')
        )

    if "deepseekcoder_6.7b" in model_loss_map and "deepseekcoder_1.3b" in model_loss_map:
        big_loss = model_loss_map["deepseekcoder_6.7b"]
        small_loss = model_loss_map["deepseekcoder_1.3b"]
        scores["deepseekcoder_6.7b_Ref"] = (
            big_loss - small_loss
            if not np.isnan(big_loss) and not np.isnan(small_loss) else float('nan')
        )

    if "codellama_13b" in model_loss_map and "codellama_7b" in model_loss_map:
        big_loss = model_loss_map["codellama_13b"]
        small_loss = model_loss_map["codellama_7b"]
        scores["codellama_13b_Ref"] = (
            big_loss - small_loss
            if not np.isnan(big_loss) and not np.isnan(small_loss) else float('nan')
        )

    if "deepseekcoder_33b" in model_loss_map and "deepseekcoder_6.7b" in model_loss_map:
        big_loss = model_loss_map["deepseekcoder_33b"]
        small_loss = model_loss_map["deepseekcoder_6.7b"]
        scores["deepseekcoder_33b_Ref"] = (
            big_loss - small_loss
            if not np.isnan(big_loss) and not np.isnan(small_loss) else float('nan')
        )
    return scores


if __name__ == '__main__':
    logger.info("运行 calc.py 中的测试用例...")
    sample_text_1 = "This is a sample text for testing."
    sample_tokens_1 = [
        {'token': 'This', 'logprob': -1.0},
        {'token': ' is', 'logprob': -0.5},
        {'token': ' a', 'logprob': -1.2},
        {'token': ' sample', 'logprob': -2.0},
        {'token': ' text', 'logprob': -1.5},
        {'token': ' for', 'logprob': -0.8},
        {'token': ' testing', 'logprob': -2.5},
        {'token': '.', 'logprob': -0.2}
    ]

    dummy_sample = {
        "text": sample_text_1,
        "logprobs": [-1.0, -0.5, -1.2, -2.0, -1.5, -0.8, -2.5, -0.2],
        "nb_logprobs": [-1.1, -0.6],  # mock
        "7b_logprobs": [-1.3, -0.7],  # mock
        "rec_Loss": 0.123,
    }

    scores_1 = calculate_all_scores(dummy_sample)
    logger.info(f"测试用例 1 - 分数: {scores_1}")