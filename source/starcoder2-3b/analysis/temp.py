import os
import json

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

SRC_DIR = '/home/yunxiang/work_may/week2/analysis'
MERGE_DIRS = ['git2401_merge', 'ts_merge']
OUT_DIRS = ['git2401_m1', 'ts_m1']

for merge_dir, out_dir in zip(MERGE_DIRS, OUT_DIRS):
    merge_path = os.path.join(SRC_DIR, merge_dir)
    out_path = os.path.join(SRC_DIR, out_dir)
    os.makedirs(out_path, exist_ok=True)
    for version in VERSION_ORDER:
        in_file = os.path.join(merge_path, f'{version}.jsonl_')
        out_file = os.path.join(out_path, f'{version}.jsonl_')
        if not os.path.exists(in_file):
            print(f'警告: {in_file} 不存在，跳过')
            continue
        selected = []
        with open(in_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                    code = obj.get('input', '')
                    if 400 <= len(code) < 500:
                        selected.append(line)
                        if len(selected) >= 200:
                            break
                except Exception as e:
                    print(f'解析错误: {e}')
        with open(out_file, 'w', encoding='utf-8') as fout:
            fout.writelines(selected)
        print(f'{out_file} 已写入 {len(selected)} 条样本')
