import logging
from pathlib import Path
import os
import json
from typing import List, Dict, Any
import sys

# 保证可以导入 sampling_strategies
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from sampling_strategies import LEVEL_TO_FEATURE
from .calc import calculate_all_scores
from .eval import fig_fpr_tpr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# 常量：源数据根目录（与 exp_maker 中保持一致）
# -----------------------------------------------------------------------------

DEFAULT_SOURCE_BASE = "/home/yunxiang/work_june/source"


class BaseSetAccessor:
    """辅助类：按照 (tag, level, index, variant) 从 *源数据目录* 直接检索样本。

    过去的实现依赖 baseset 中复制好的样本 jsonl 文件，造成磁盘占用与 I/O 开销巨大。
    现改为：

        • baseset_dir 仅保存 index_tag_<level>.jsonl 等索引文件，便于 attempt_maker 采样。
        • 真正的样本数据始终存放于 source_base (默认 /home/yunxiang/work_june/source)。

    因此，本类初始化时既接受 baseset_dir（可选，仅为兼容旧接口），也接受 source_base。
    _resolve_file_path 将仅依赖 source_base 来定位文件。
    """

    def __init__(self, baseset_dir: str = "", source_base: str = DEFAULT_SOURCE_BASE):
        # index 根目录（可能为空，仅用于外部兼容）
        self.index_root = Path(baseset_dir) if baseset_dir else None

        # 样本数据根目录
        self.source_base = Path(source_base)
        assert self.source_base.exists(), f"source_base 不存在: {self.source_base}"

        # cache: (file_path) -> {index: obj}
        self._file_cache: Dict[Path, Dict[int, Dict[str, Any]]] = {}

        # level -> feature 缓存
        self.level2feature = LEVEL_TO_FEATURE

    # ------------------------- 内部辅助 -------------------------
    # 支持的模型前缀与其目录名映射
    _MODEL_DIRS = {
        "starcoder2_3b": "starcoder2-3b",
        "starcoder2_7b": "starcoder2-7b",
        "deepseekcoder_1.3b": "deepseekcoder-1.3b",
        "deepseekcoder_6.7b": "deepseekcoder-6.7b",
        "santacoder_1.1b": "santacoder-1.1b",
        "codellama_7b": "codellama-7b",
        "codellama_13b": "codellama-13b",
        "deepseekcoder_33b": "deepseekcoder-33b",
    }

    # 根据模型动态生成 variant 模板
    _VARIANT_TEMPLATE: Dict[str, str] = {}
    for _key, _dir in _MODEL_DIRS.items():
        # 原始 & 邻域
        _VARIANT_TEMPLATE[_key] = f"{_dir}/analysis/{{base}}"  # origin
        _VARIANT_TEMPLATE[f"{_key}_nb"] = f"{_dir}/analysis/{{base}}_nb"  # neighbor
        # rec_new
        _VARIANT_TEMPLATE[f"{_key}_rec_new"] = f"{_dir}/analysis_rec_new/{{base}}"

    del _key, _dir  # 清理局部变量

    def _resolve_file_path(self, tag: int, level_code: str, variant: str = "origin") -> Path:
        """解析 (tag, level_code, variant) 对应的文件绝对路径。

        与旧实现区别：不再在 baseset_dir 中查找，而是直接使用 source_base 下的原始分析文件。
        """
        if variant not in self._VARIANT_TEMPLATE:
            raise ValueError(f"未知的 variant: {variant}")

        # tag -> memall / nmeall
        base_name = "memall" if tag == 1 else "nmeall"
        variant_rel = self._VARIANT_TEMPLATE[variant].format(base=base_name)

        feature = self.level2feature[level_code]

        dir_path = self.source_base / variant_rel
        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        for fname in os.listdir(dir_path):
            if feature in fname and fname.endswith(".jsonl_"):
                return dir_path / fname

        raise FileNotFoundError(
            f"在 {dir_path} 中找不到包含特征 {feature} 的文件 (variant={variant})。")

    def _load_file_to_cache(self, fpath: Path):
        """将文件加载到缓存 (一次性)。"""
        mapping: Dict[int, Dict[str, Any]] = {}
        with open(fpath, "r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                    idx = obj.get("index")
                    if idx is not None:
                        mapping[int(idx)] = obj
                except Exception:
                    continue
        self._file_cache[fpath] = mapping

    def fetch(self, tag: int, level_code: str, idx: int, variant: str = "origin") -> Dict[str, Any]:
        """根据 variant 返回样本字典。"""
        fpath = self._resolve_file_path(tag, level_code, variant)
        if fpath not in self._file_cache:
            self._load_file_to_cache(fpath)
        sample = self._file_cache[fpath].get(int(idx))
        if sample is None:
            raise KeyError(f"在文件 {fpath} 中未找到 index={idx}")
        return sample


# -----------------------------------------------------------------------------
# 通用工具


def _get_log_probs_from_tokens(precomputed_tokens: Any) -> List[float]:
    """从 token 列表中抽取 logprob 数值 (List[float])。

    tokens 的格式应为 List[Dict]，其中每个元素含有字段 'logprob'。若格式不符，返回空列表。
    """
    log_probs: List[float] = []
    if not isinstance(precomputed_tokens, list):
        return log_probs
    for item in precomputed_tokens:
        if isinstance(item, dict) and "logprob" in item:
            try:
                log_probs.append(float(item["logprob"]))
            except (TypeError, ValueError):
                continue
    return log_probs


# -----------------------------------------------------------------------------
# 推理与评估


def _build_samples(raw_records: List[Dict[str, Any]], accessor: BaseSetAccessor) -> List[Dict[str, Any]]:
    """根据 index/tag/level/label 构造包含多模型 logprobs / rec_new_Loss 的样本。"""
    processed: List[Dict[str, Any]] = []

    model_keys = list(BaseSetAccessor._MODEL_DIRS.keys())

    for rec in raw_records:
        try:
            idx: int = rec["index"]
            tag: int = rec["tag"]
            level_code: str = rec["level"]
            label_val: int = rec["label"]

            # 统一使用第一个模型的文本作为 text（各模型 input 应一致）
            first_origin = accessor.fetch(tag, level_code, idx, variant=model_keys[0])
            text_val = first_origin.get("input")

            sample_dict: Dict[str, Any] = {
                "text": text_val,
                "label": label_val,
            }

            for mkey in model_keys:
                try:
                    origin_obj = accessor.fetch(tag, level_code, idx, variant=mkey)
                    nb_obj = accessor.fetch(tag, level_code, idx, variant=f"{mkey}_nb")
                    rec_new_obj = accessor.fetch(tag, level_code, idx, variant=f"{mkey}_rec_new")

                    sample_dict[f"{mkey}_logprobs"] = _get_log_probs_from_tokens(origin_obj.get("tokens"))
                    sample_dict[f"{mkey}_nb_logprobs"] = _get_log_probs_from_tokens(nb_obj.get("tokens"))
                    sample_dict[f"{mkey}_rec_new_Loss"] = rec_new_obj.get("Loss", rec_new_obj.get("loss"))
                except Exception as inner_e:
                    logger.warning(f"模型 {mkey} 样本(index={idx}) 获取失败: {inner_e}")

            processed.append(sample_dict)
        except Exception as e:
            logger.warning(f"跳过样本 (index={rec.get('index')})，原因: {e}")
    return processed


def _inference(ex: Dict[str, Any]) -> Dict[str, Any]:
    """针对单个样本计算分数，结果写入 ex['pred']"""
    try:
        ex["pred"] = calculate_all_scores(ex)
    except Exception as e:
        logger.error(f"计算分数时出错: {e}")
        ex["pred"] = {"error": str(e)}
    return ex


def _evaluate(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对全部样本逐一推理，无进度条输出（避免嵌套 tqdm）。"""
    logger.info(f"开始评估，共 {len(data)} 条样本……")
    output = []
    for ex in data:
        output.append(_inference(ex))
        # print(output)
        # return
    return output


# -----------------------------------------------------------------------------
# 对外主函数

def process_jsonl(dataset_path: str, baseset_dir: str, output_dir: str, *, accessor: "BaseSetAccessor | None" = None):
    """给定 attempt 的某个测试数据集，输出评估结果到 output_dir"""
    dataset_path = str(dataset_path)
    baseset_dir = str(baseset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"处理数据集: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as fin:
        raw_records = [json.loads(line) for line in fin]

    # 允许外部复用同一 accessor 以减少重复 I/O
    if accessor is None:
        accessor = BaseSetAccessor(baseset_dir)

    samples = _build_samples(raw_records, accessor)
    if not samples:
        logger.error("未能构造任何可用样本，跳过。")
        return

    all_output = _evaluate(samples)

    # # 保存预测
    # pred_path = output_dir / "predictions.jsonl"
    # with open(pred_path, "w", encoding="utf-8") as fout:
    #     for ex in all_output:
    #         fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
    # logger.info(f"已保存预测结果到 {pred_path}")

    # 生成 ROC 图 & AUC
    try:
        fig_fpr_tpr(all_output, str(output_dir))
        logger.info("已生成评估图表及 AUC 文本。")
    except Exception as e:
        logger.error(f"生成评估图表失败: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="对单个 jsonl 数据集进行评估")
    parser.add_argument("--data", required=True, help="待评估的数据集 jsonl")
    parser.add_argument("--baseset_dir", required=True, help="对应的 baseset 目录 (如 baseset/exp1)")
    parser.add_argument("--output_dir", required=True, help="结果输出目录")
    args = parser.parse_args()

    process_jsonl(args.data, args.baseset_dir, args.output_dir) 