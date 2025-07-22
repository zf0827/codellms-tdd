import os
import requests
import subprocess
import tempfile
import json
import ast
from pathlib import Path

# --- 配置 ---
# 建议使用个人访问令牌以获取更高的API速率限制。
# 请在您的环境中设置 GITHUB_TOKEN 环境变量。
# 例如: export GITHUB_TOKEN='your_personal_access_token'
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

# 仓库大小阈值，单位为KB (100MB = 100,000KB)
MAX_REPO_SIZE_KB = 50000

# --- 第 1 步: 获取候选仓库 ---

def find_repositories(query="language:python created:>2024-01-01 stars:<2000", sort="stars", order="desc", per_page=1):
    """
    使用 GitHub API 根据查询条件查找仓库。
    """
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "sort": sort, "order": order, "per_page": per_page}
    print(f"正在使用查询向 GitHub API 发出请求: {query}")
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()["items"]

# --- 第 2 步: 分析仓库提交历史 ---

def clone_repo(clone_url, local_path):
    """
    将远程仓库克隆到本地。
    """
    print(f"正在克隆 {clone_url} 到 {local_path}...")
    subprocess.run(["git", "clone", clone_url, local_path], check=True, capture_output=True)

def get_initial_commits(repo_path):
    """
    获取仓库的初始提交（没有父提交的提交）。
    """
    result = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        cwd=repo_path, check=True, capture_output=True, text=True
    )
    return result.stdout.strip().split("\n")

def get_relevant_commits(repo_path, initial_commit_hashes):
    """
    获取初始提交之后的所有修改了 .py 文件的、非合并的提交。
    """
    if not initial_commit_hashes or not initial_commit_hashes[0]:
        return []
    initial_commit_hash = initial_commit_hashes[0]
    command = [
        "git", "log", f"{initial_commit_hash}..HEAD",
        "--pretty=format:%H", "--no-merges", "--", "*.py"
    ]
    result = subprocess.run(command, cwd=repo_path, check=True, capture_output=True, text=True)
    if result.stdout:
        return result.stdout.strip().split("\n")
    return []

# --- 第 3 步: 提取代码变更信息 ---

def get_changed_lines_for_commit(repo_path, commit_hash):
    """
    对于给定的提交，获取所有修改过的Python文件及其变更的行号区间。
    """
    command = ["git", "show", commit_hash, "--unified=0", "--", "*.py"]
    result = subprocess.run(command, cwd=repo_path, check=True, capture_output=True, text=True, errors='ignore')
    
    changed_files = {}
    current_file = None
    for line in result.stdout.split('\n'):
        if line.startswith('+++ b/'):
            current_file = line[6:]
            if current_file.endswith('.py'):
                changed_files[current_file] = []
            else:
                current_file = None
        elif line.startswith('@@') and current_file:
            parts = line.split(' ')
            if len(parts) > 2:
                hunk_info = parts[2]
                if hunk_info.startswith('+'):
                    try:
                        if ',' in hunk_info:
                            start, count = map(int, hunk_info[1:].split(','))
                        else:
                            start = int(hunk_info[1:])
                            count = 1
                        if count > 0:
                            changed_files[current_file].append((start, start + count - 1))
                    except ValueError:
                        print(f"无法解析 hunk: {line} in commit {commit_hash}")
                        continue

    # 合并重叠或相邻的区间
    for file_path, intervals in changed_files.items():
        if not intervals: continue
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        changed_files[file_path] = merged
        
    return {f: r for f, r in changed_files.items() if r}

# --- 第 4 步: 映射变更到函数并提取 ---

class FunctionVisitor(ast.NodeVisitor):
    """一个AST访问者，用于查找与变更行号重叠的函数。"""
    def __init__(self, changed_line_ranges):
        self.changed_line_ranges = changed_line_ranges
        self.found_functions = []

    def _check_overlap(self, node):
        func_start = node.lineno
        func_end = getattr(node, 'end_lineno', func_start)
        for change_start, change_end in self.changed_line_ranges:
            if max(func_start, change_start) <= min(func_end, change_end):
                self.found_functions.append(node)
                return

    def visit_FunctionDef(self, node):
        self._check_overlap(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._check_overlap(node)
        self.generic_visit(node)

def extract_functions_from_changes(repo_path, commit_hash, changed_files_with_lines):
    """
    检出文件、执行AST分析并提取受影响的函数。
    """
    extracted_functions = []
    for file_path, line_ranges in changed_files_with_lines.items():
        try:
            # 使用 `git show` 获取文件内容，避免修改工作目录
            file_content = subprocess.run(
                ["git", "show", f"{commit_hash}:{file_path}"],
                cwd=repo_path, check=True, capture_output=True, text=True, errors='ignore'
            ).stdout
        except subprocess.CalledProcessError:
            print(f"无法读取文件 {file_path} 在提交 {commit_hash}。跳过。")
            continue

        try:
            tree = ast.parse(file_content, filename=file_path)
            visitor = FunctionVisitor(line_ranges)
            visitor.visit(tree)
            
            for func_node in visitor.found_functions:
                # 使用 ast.get_source_segment 安全地提取函数源代码 (需要 Python 3.8+)
                source_code = ast.get_source_segment(file_content, func_node)
                if source_code:
                    extracted_functions.append({
                        "repository_name": None, # 稍后填充
                        "repository_url": None, # 稍后填充
                        "commit_hash": commit_hash,
                        "file_path": file_path,
                        "function_name": func_node.name,
                        "input": source_code,
                    })
        except SyntaxError:
            print(f"在 {file_path} (commit {commit_hash}) 中存在语法错误。跳过。")
        except Exception as e:
            print(f"在AST分析 {file_path} (commit {commit_hash}) 时发生未知错误: {e}")
            
    return extracted_functions

# --- 主流程控制 ---

def is_repo_too_large(repo_info):
    """
    检查仓库大小是否超过阈值。
    
    GitHub API中的size字段以KB为单位。
    """
    repo_size = repo_info.get("size", 0)
    if repo_size > MAX_REPO_SIZE_KB:
        print(f"仓库 {repo_info['name']} 大小为 {repo_size/1000:.2f}MB，超过了 {MAX_REPO_SIZE_KB/1000:.2f}MB 的阈值，跳过处理。")
        return True
    return False

def process_repository(repo_info):
    """
    处理单个仓库的完整流程。
    """
    clone_url = repo_info["clone_url"]
    repo_name = repo_info["name"]
    repo_html_url = repo_info["html_url"]
    all_novel_functions = []
    
    # 检查仓库大小
    if is_repo_too_large(repo_info):
        return []

    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / repo_name
        try:
            clone_repo(clone_url, repo_path)
            initial_commits = get_initial_commits(repo_path)
            relevant_commits = get_relevant_commits(repo_path, initial_commits)
            
            print(f"为仓库 {repo_name} 找到 {len(relevant_commits)} 个相关提交。")

            for i, commit_hash in enumerate(relevant_commits):
                print(f"  正在处理提交 {i+1}/{len(relevant_commits)}: {commit_hash[:7]}...", end='\r')
                changed_files = get_changed_lines_for_commit(repo_path, commit_hash)
                if not changed_files:
                    continue
                
                functions = extract_functions_from_changes(repo_path, commit_hash, changed_files)
                for func_data in functions:
                    func_data["repository_name"] = repo_name
                    func_data["repository_url"] = repo_html_url
                    all_novel_functions.append(func_data)
            print() # 换行

        except subprocess.CalledProcessError as e:
            print(f"处理 {repo_name} 时Git命令失败: {e.stderr}")
            return []
        except Exception as e:
            print(f"处理 {repo_name} 时发生未知错误: {e}")
            return []
    
    if all_novel_functions:
        print(f"从仓库 {repo_name} 成功提取 {len(all_novel_functions)} 个函数。")
    
    return all_novel_functions

def main():
    """
    脚本主入口。
    """
    if not GITHUB_TOKEN:
        print("错误: 环境变量 GITHUB_TOKEN 未设置。")
        print("请在GitHub上创建一个个人访问令牌并设置该变量。")
        return

    print("--- 开始基于Git历史的新颖函数提取流程 ---")
    
    NUM_REPOS_TO_PROCESS = 7
    all_functions_from_all_repos = []

    try:
        repositories = find_repositories(per_page=NUM_REPOS_TO_PROCESS)
        if not repositories:
            print("根据指定条件，未找到任何仓库。")
            return
        
        print(f"找到 {len(repositories)} 个仓库进行处理。")

        for repo_info in repositories:
            print(f"\n--- 开始处理仓库: {repo_info['full_name']} ({repo_info['stargazers_count']} ★) ---")
            functions_from_repo = process_repository(repo_info)
            all_functions_from_all_repos.extend(functions_from_repo)

    except requests.HTTPError as e:
        print(f"从GitHub获取仓库失败: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"发生未知错误: {e}")

    # 最终写入步骤
    if all_functions_from_all_repos:
        output_filename = "novel_functions.jsonl"
        with open(output_filename, "w", encoding="utf-8") as f:
            for func in all_functions_from_all_repos:
                # 重新排序键以将 'input' 放在首位
                output_record = {
                    "input": func["input"],
                    "repository_name": func["repository_name"],
                    "repository_url": func["repository_url"],
                    "commit_hash": func["commit_hash"],
                    "file_path": func["file_path"],
                    "function_name": func["function_name"],
                }
                f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
        print(f"\n✅ 全部完成! 已提取 {len(all_functions_from_all_repos)} 个函数到 {output_filename}")
    else:
        print(f"\n处理完成，但没有提取到任何函数。")

if __name__ == "__main__":
    main()
