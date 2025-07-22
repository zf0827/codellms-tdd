import argparse
from pathlib import Path
import sys
from tqdm import tqdm

# 允许导入 src 包
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.run import process_jsonl, BaseSetAccessor

# python3 result_maker.py --exp_name exp1 --attempt_id 1
# python3 result_maker.py --exp_name exp2 --attempt_id 1
def main():
    parser = argparse.ArgumentParser(description="批量运行 attempt 目录下所有测试数据集并生成结果")
    parser.add_argument("--exp_name", required=True, help="实验名称 (如 exp1)")
    parser.add_argument("--attempt_id", type=int, default=1, help="attempt 序号 (默认 1)")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="若结果目录下已存在 predictions.jsonl，则跳过该数据集 (增量式)"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    dataset_root = base_dir / "dataset" / args.exp_name / f"attempt{args.attempt_id}"
    baseset_dir = base_dir / "baseset" / args.exp_name
    result_root = base_dir / "result" / args.exp_name / f"attempt{args.attempt_id}"

    if not dataset_root.exists():
        raise FileNotFoundError(f"找不到数据集目录: {dataset_root}")
    if not baseset_dir.exists():
        raise FileNotFoundError(f"找不到 baseset 目录: {baseset_dir}")

    jsonl_files = list(dataset_root.rglob("*.jsonl"))
    if not jsonl_files:
        print("未在数据集目录中找到任何 .jsonl 文件，退出。")
        return

    print(f"共发现 {len(jsonl_files)} 个数据集文件，开始处理……")

    # 预创建一个共享 accessor，避免在每个数据集处理时重复解析大文件
    shared_accessor = BaseSetAccessor(str(baseset_dir))

    for jf in tqdm(jsonl_files, desc="Running datasets"):
        rel_path = jf.relative_to(dataset_root)
        out_dir = result_root / rel_path.parent / rel_path.stem

        # 若启用跳过逻辑且结果已存在，则继续下一文件
        auc_file = out_dir / "auc.txt"
        if args.skip_existing and auc_file.exists():
            tqdm.write(f"[SKIP] 已存在结果: {auc_file}")
            continue

        try:
            process_jsonl(str(jf), str(baseset_dir), str(out_dir), accessor=shared_accessor)
        except Exception as e:
            print(f"处理 {jf} 时出错: {e}")

        # return

    print("全部数据集处理完毕！")


if __name__ == "__main__":
    main()
