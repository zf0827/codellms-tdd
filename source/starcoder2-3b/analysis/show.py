import os
import json
import matplotlib.pyplot as plt

# 版本顺序
VERSION_ORDER = [
    'original',
    'level1',
    'level2_1',
    'level2_2',
    'level2_3',
    'level3_sim0.9',
    'level3_sim0.7',
    'level3_sim0.5',
    'level3_0.8RE'
]

MERGE_DIRS = ['git2401_merge', 'ts_merge']
SRC_DIR = '/home/yunxiang/work_may/week2/analysis'

# 统计区间
bins = list(range(100, 850, 50))

for merge_dir in MERGE_DIRS:
    merge_path = os.path.join(SRC_DIR, merge_dir)
    for version in VERSION_ORDER:
        file_path = os.path.join(merge_path, f'{version}.jsonl_')
        if not os.path.exists(file_path):
            print(f'警告: {file_path} 不存在，跳过')
            continue
        lengths = []
        with open(file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                    code = obj.get('input', '')
                    lengths.append(len(code))
                except Exception as e:
                    print(f'解析错误: {e}')
        # 绘图
        plt.figure(figsize=(8, 5))
        plt.hist(lengths, bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel('代码长度', fontsize=13)
        plt.ylabel('样本数', fontsize=13)
        plt.title(f'{merge_dir} - {version} 代码长度分布', fontsize=15)
        plt.xticks(bins)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        out_img = os.path.join(merge_path, f'{merge_dir}_{version}_length_hist.png')
        plt.tight_layout()
        plt.savefig(out_img)
        plt.close()
        print(f'已保存: {out_img}')
