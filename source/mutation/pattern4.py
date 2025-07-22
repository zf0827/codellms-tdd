# coding: utf-8
"""
模块名称: pattern4.py
模式识别: 等于比较语句 (例如: x == y)
扰动策略: 将原始的等于比较语句转换为使用 'not' 和 '!=' 的等价形式。
    原始形式:
        x == y
    修改后形式:
        not (x != y)
使用方法:
    此模块主要通过 `apply_perturbation_equality_to_not_in_equality` 函数被扰动框架 (如 mutaor3.py) 调用。
    调用时需要传入待扰动的 Python 代码的 AST 根节点以及一个 `threshold_ratio` 参数。
    `threshold_ratio` (0.0 到 1.0) 控制符合条件的比较语句中被实际扰动的比例。
    例如，`threshold_ratio = 0.5` 表示大约一半的合格比较语句会被修改。
    内部使用 `EqualityToNotInEqualityTransformer` (一个 `ast.NodeTransformer` 子类) 来执行实际的 AST 转换。
    重要潜在隐患:
    此转换依赖于被比较对象的 `__eq__` 和 `__ne__` 方法遵循标准的逻辑关系,
    即 `(obj1 == obj2)` 在逻辑上等价于 `not (obj1 != obj2)`。
    如果一个类自定义了 `__eq__` 和 `__ne__` 方法，并且它们不遵循这种关系
    （例如，`obj1 != obj2` 的实现不是 `not (obj1 == obj2)`），
    则此转换可能会改变原始代码的语义。此策略主要适用于比较行为符合 Python 
    标准约定的对象。同时，此策略仅适用于简单的、非链式的比较，例如 `a == b`，
    而不适用于链式比较如 `a == b == c`。
"""
import ast
import random

class EqualityToNotInEqualityTransformer(ast.NodeTransformer):
    """
    AST转换器，用于将选定的 '==' 比较操作修改为 'not (!=)'。
    原始: x == y
    修改后: not (x != y)
    """
    def __init__(self, nodes_to_transform):
        super().__init__()
        self.nodes_to_transform = set(nodes_to_transform)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        # 确保是我们想要转换的节点，并且是单个比较操作 (如 a == b, 而非 a == b == c)
        # 并且操作符是 ast.Eq
        if node in self.nodes_to_transform and \
           len(node.ops) == 1 and \
           isinstance(node.ops[0], ast.Eq) and \
           len(node.comparators) == 1:

            original_left = node.left
            original_comparator = node.comparators[0]

            # 1. 创建内部的 'x != y' 比较节点
            # 我们需要确保原始节点的子节点 (left, comparators[0]) 被正确地传递
            # 并且它们上下文 (ctx) 应该是 ast.Load()，这在比较中通常是默认的
            not_eq_compare_node = ast.Compare(
                left=original_left,  # 原始左操作数
                ops=[ast.NotEq()],   # 操作符变为 NotEq
                comparators=[original_comparator] # 原始右操作数
            )
            # 从原始比较节点复制位置信息给新的内部比较节点
            ast.copy_location(not_eq_compare_node, node)

            # 2. 创建 'not (...)' 一元操作节点
            new_unary_op_node = ast.UnaryOp(
                op=ast.Not(),
                operand=not_eq_compare_node
            )
            # 将整个新表达式 (UnaryOp) 的位置信息设置为与原始 '==' 表达式相同
            ast.copy_location(new_unary_op_node, node)
            
            # 递归填充所有新创建的节点及其子节点可能缺失的源码位置属性
            return ast.fix_missing_locations(new_unary_op_node)

        # 对于不转换的节点或不符合条件的 Compare 节点，继续访问其子节点
        # 以便转换器可以处理嵌套结构中的其他可能转换
        return self.generic_visit(node)

def apply_perturbation_equality_to_not_in_equality(ast_root: ast.AST, threshold_ratio: float = 1.0) -> ast.AST:
    """
    应用 "等于布尔表达式的等价替换 (x == y  ->  not (x != y))" 扰动策略。

    重要潜在隐患:
    此转换依赖于被比较对象的 `__eq__` 和 `__ne__` 方法遵循标准的逻辑关系,
    即 `(obj1 == obj2)` 在逻辑上等价于 `not (obj1 != obj2)`。
    如果一个类自定义了 `__eq__` 和 `__ne__` 方法，并且它们不遵循这种关系
    （例如，`obj1 != obj2` 的实现不是 `not (obj1 == obj2)`），
    则此转换可能会改变原始代码的语义。此策略主要适用于比较行为符合 Python 
    标准约定的对象。

    Args:
        ast_root: 要转换的AST的根节点。
        threshold_ratio (float): 要扰动的已识别合格 'Compare' 语句的比例 (0.0 到 1.0)。
                                 1.0 表示全部扰动，0.5 表示扰动50%，等等。

    Returns:
        修改后的AST根节点。
    """
    candidate_compare_nodes = []
    for node in ast.walk(ast_root):
        if isinstance(node, ast.Compare):
            # 筛选条件：
            # 1. 只有一个操作符 (ops 列表长度为1)
            # 2. 该操作符是 ast.Eq (==)
            # 3. 只有一个比较对象 (comparators 列表长度为1)
            #    这排除了链式比较 a == b == c
            if len(node.ops) == 1 and \
               isinstance(node.ops[0], ast.Eq) and \
               len(node.comparators) == 1:
                candidate_compare_nodes.append(node)

    if not candidate_compare_nodes:
        return ast_root # 没有可扰动的 '==' 比较语句

    num_to_perturb = int(len(candidate_compare_nodes) * threshold_ratio)
    
    nodes_to_actually_transform = []
    if num_to_perturb > 0:
        # random.sample 的 k 参数必须非负且不大于总体数量
        actual_k = min(num_to_perturb, len(candidate_compare_nodes))
        if actual_k > 0 :
            nodes_to_actually_transform = random.sample(candidate_compare_nodes, actual_k)
    
    if not nodes_to_actually_transform:
        return ast_root # 没有选择任何节点进行扰动

    transformer = EqualityToNotInEqualityTransformer(nodes_to_transform=nodes_to_actually_transform)
    modified_ast = transformer.visit(ast_root) # visit 返回修改后的根节点
    
    # 确保整个修改后的AST所有节点都有位置信息
    return ast.fix_missing_locations(modified_ast) 

def count_perturbation_candidates(ast_root):
    count = 0
    for node in ast.walk(ast_root):
        if isinstance(node, ast.Compare):
            if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq) and len(node.comparators) == 1:
                count += 1
    return count 