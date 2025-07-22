# coding: utf-8
"""
模块名称: pattern2.py
模式识别: IF 语句 (ast.If 节点)
扰动策略: 将原始 IF 语句的条件修改为 'condition and True'。
    原始形式:
        if condition:
            ...
    修改后形式:
        if condition and True:
            ...
使用方法:
    此模块主要通过 `apply_perturbation_if_and_true` 函数被扰动框架 (如 mutaor3.py) 调用。
    调用时需要传入待扰动的 Python 代码的 AST 根节点以及一个 `threshold_ratio` 参数。
    `threshold_ratio` (0.0 到 1.0) 控制符合条件的 IF 语句中被实际扰动的比例。
    例如，`threshold_ratio = 0.5` 表示大约一半的 IF 语句条件会被修改。
    内部使用 `IfConditionAndTrueTransformer` (一个 `ast.NodeTransformer` 子类) 来执行实际的 AST 转换。
    辅助类 `NodeFinder` (在 ast_utils.py 中定义，但此模块直接使用其结果) 用于查找所有候选的 ast.If 节点。
"""
import ast
import random
from ast_utils import NodeFinder

# --- Perturbation Strategy: Mode 2 (If statement: if condition => if condition and True) ---

class IfConditionAndTrueTransformer(ast.NodeTransformer):
    """
    一个AST转换器，用于将选定的 'if' 语句的条件修改为 'condition and True'。
    """
    def __init__(self, nodes_to_transform):
        super().__init__()
        # 使用set以便快速查找
        self.nodes_to_transform = set(nodes_to_transform)

    def visit_If(self, node):
        original_node = node
        if original_node in self.nodes_to_transform:
            original_condition = original_node.test

            # 创建 'True' 常量节点
            true_constant = ast.Constant(value=True)
            # 尝试从原始条件复制位置信息给新节点
            ast.copy_location(true_constant, original_condition)

            # 创建 'condition and True' AST节点
            new_condition = ast.BoolOp(
                op=ast.And(),
                values=[original_condition, true_constant]
            )
            ast.copy_location(new_condition, original_condition)

            # 修改现有节点
            node.test = new_condition
            
            # 确保新生成的节点部分有位置信息
            ast.fix_missing_locations(node)

        # 无论是否转换当前节点，都继续访问其子节点
        return self.generic_visit(node)


def apply_perturbation_if_and_true(ast_root, threshold_ratio=1.0):
    """
    应用"if condition => if condition and True"扰动策略。

    Args:
        ast_root: 要转换的AST的根节点。
        threshold_ratio (float): 要扰动的已识别'if'语句的比例 (0.0 到 1.0)。
                                 1.0 表示全部扰动，0.5 表示扰动50%，等等。

    Returns:
        修改后的AST根节点。
    """
    if_finder = NodeFinder(ast.If)
    candidate_if_nodes = if_finder.find(ast_root)

    if not candidate_if_nodes:
        return ast_root # 没有可扰动的if语句

    # 根据阈值确定要扰动的节点数量
    num_to_perturb = int(len(candidate_if_nodes) * threshold_ratio)
    
    # 从候选中随机选择要实际转换的节点
    # random.sample确保选择不重复的元素
    if num_to_perturb > 0 : # random.sample k must be non-negative and not larger than population
        nodes_to_actually_transform = random.sample(candidate_if_nodes, num_to_perturb)
    else:
        nodes_to_actually_transform = []


    if not nodes_to_actually_transform:
        return ast_root # 没有选择任何节点进行扰动

    # 创建并应用转换器
    transformer = IfConditionAndTrueTransformer(nodes_to_transform=nodes_to_actually_transform)
    modified_ast = transformer.visit(ast_root)
    
    # visit方法可以返回一个新树，或者修改并返回原始树。
    # NodeTransformer通常返回一个新树或修改后的同一棵树。
    # ast.fix_missing_locations 确保所有新创建的节点都有行号等信息，这对于后续unparse很重要。
    return ast.fix_missing_locations(modified_ast) 

def count_perturbation_candidates(ast_root):
    count = 0
    for node in ast.walk(ast_root):
        if isinstance(node, ast.If):
            count += 1
    return count