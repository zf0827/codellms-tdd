import argparse
import os
import sys

# python3 mutation.py --input_file /home/yunxiang/work_may/week2/dataset/git2401_p3/original.jsonl --output_dir /home/yunxiang/work_may/week2/dataset/git2401_p3 --output_name git2401_p3
def main():
    parser = argparse.ArgumentParser(description="批量生成多等级扰动数据集")
    parser.add_argument("--input_file", required=True, help="输入JSONL文件路径")
    parser.add_argument("--output_dir", required=True, help="输出文件夹")
    parser.add_argument("--output_name", required=True, help="输出文件基础名（不带后缀）")
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir
    output_name = args.output_name

    os.makedirs(output_dir, exist_ok=True)

    # 1. 等级一：mutaor1.py
    out1 = os.path.join(output_dir, output_name + "_level1.jsonl")
    cmd1 = f"python3 mutaor1.py --input_file {input_file} --output_file {out1}"

    # 2. 等级二：rewriter.py（会生成3个文件）
    rewriter_py = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../LLM_rewrite/rewriter.py'))
    out2_base = os.path.join(output_dir, output_name + "_level2")
    out2_1 = out2_base + "_1.jsonl"
    out2_2 = out2_base + "_2.jsonl"
    out2_3 = out2_base + "_3.jsonl"
    cmd2 = f"python3 {rewriter_py} --input_file {input_file} --output_file {out2_base}"
    # rewriter会生成 _level2_1.jsonl, _level2_2.jsonl, _level2_3.jsonl

    # 2.1 对_level2_1和_level2_2执行clean.py
    clean_py = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/clean.py'))
    cmd_clean_1 = f"python3 {clean_py} --file {out2_1}"
    cmd_clean_2 = f"python3 {clean_py} --file {out2_2}"

    # 3. 等级三：mutaor3.py，similarity=0.5,0.7,0.9,0.8RE
    out3_05 = os.path.join(output_dir, output_name + "_level3_sim0.5.jsonl")
    out3_07 = os.path.join(output_dir, output_name + "_level3_sim0.7.jsonl")
    out3_09 = os.path.join(output_dir, output_name + "_level3_sim0.9.jsonl")
    out3_08_RE = os.path.join(output_dir, output_name + "_level3_0.8RE.jsonl")
    cmd3_05 = f"python3 mutaor3.py --input_file {input_file} --output_file {out3_05} --similarity 0.5"
    cmd3_07 = f"python3 mutaor3.py --input_file {input_file} --output_file {out3_07} --similarity 0.7"
    cmd3_09 = f"python3 mutaor3.py --input_file {input_file} --output_file {out3_09} --similarity 0.9"
    cmd3_08_RE = f"python3 mutaor3.py --input_file {out2_1} --output_file {out3_08_RE} --similarity 0.8"

    # 定义每个命令对应的输出文件（有的命令有多个输出文件）
    cmd_outputs = [
        (cmd1, [out1]),
        (cmd2, [out2_1, out2_2, out2_3]),
        (cmd_clean_1, [out2_1]),  # clean.py会覆盖原文件
        (cmd_clean_2, [out2_2]),
        (cmd3_05, [out3_05]),
        (cmd3_07, [out3_07]),
        (cmd3_09, [out3_09]),
        (cmd3_08_RE, [out3_08_RE]),
    ]

    for cmd, outs in cmd_outputs:
        # 检查所有输出文件是否都已存在
        all_exist = all([os.path.exists(out) for out in outs])
        if all_exist:
            print(f"跳过命令（输出文件已存在）: {cmd}")
            continue
        print(f"运行命令: {cmd}")
        ret = os.system(cmd)
        if ret != 0:
            print(f"命令执行失败: {cmd}")
            sys.exit(1)
    print("全部扰动数据集已生成！")

if __name__ == "__main__":
    main()
