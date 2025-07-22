# coding: utf-8
"""
模块名称: pattern1.py
模式识别: 简单赋值语句 (例如: x = expression)
扰动策略: 将原始的简单赋值语句转换为使用临时变量的两条语句。
    原始形式:
        x = expression
    修改后形式:
        _p_temp_val_N = expression  # N 是一个递增的唯一编号
        x = _p_temp_val_N
使用方法:
    此模块主要通过 `apply_perturbation_assignment_temp_var` 函数被扰动框架 (如 mutaor3.py) 调用。
    调用时需要传入待扰动的 Python 代码的 AST 根节点以及一个 `threshold_ratio` 参数。
    `threshold_ratio` (0.0 到 1.0) 控制符合条件的赋值语句中被实际扰动的比例。
    例如，`threshold_ratio = 0.5` 表示大约一半的简单赋值语句会被修改。
    内部使用 `AssignmentToTempVarTransformer` (一个 `ast.NodeTransformer` 子类) 来执行实际的 AST 转换。
    临时变量名会自动生成并确保唯一性（在单次 `apply_perturbation_assignment_temp_var` 调用内，
    通过 Transformer 实例的类变量 `_temp_var_counter` 实现）。
    可以通过 `AssignmentToTempVarTransformer.reset_counter()` 来重置计数器，主要用于测试。
"""
import ast
import random

class AssignmentToTempVarTransformer(ast.NodeTransformer):
    """
    一个AST转换器，用于将选定的 'Assign' 语句修改为使用临时变量。
    原始: x = expression
    修改后:
        _p_temp_val_N = expression
        x = _p_temp_val_N
    """
    _temp_var_counter = 0  # 类变量，用于生成唯一的临时变量名后缀

    def __init__(self, nodes_to_transform):
        super().__init__()
        # 使用set以便快速查找
        self.nodes_to_transform = set(nodes_to_transform)

    @classmethod
    def _generate_temp_var_name(cls):
        """生成一个唯一的临时变量名。"""
        cls._temp_var_counter += 1
        return f"_p_temp_val_{cls._temp_var_counter}"

    @classmethod
    def reset_counter(cls):
        """重置临时变量计数器 (主要用于测试目的)。"""
        cls._temp_var_counter = 0

    def visit_Assign(self, node: ast.Assign) -> list[ast.AST] | ast.AST:
        """
        访问 Assign 节点。
        如果节点在 self.nodes_to_transform 中且为简单赋值，
        则将其替换为两条使用临时变量的赋值语句。
        """
        # 首先调用 generic_visit 不是必需的，因为我们要么替换节点，要么原样返回。
        # node = super().generic_visit(node) # 如果需要先转换子节点，可以取消注释

        if node in self.nodes_to_transform:
            # 仅处理简单赋值，例如: name = val, obj.attr = val, obj[item] = val
            # 不处理元组解包等复杂赋值: a, b = 1, 2
            if len(node.targets) == 1: #  and isinstance(node.targets[0], (ast.Name, ast.Attribute, ast.Subscript)):
                                        # 检查 target 类型可以增加稳健性，但用户未明确要求，暂时简化
                
                temp_var_name = self._generate_temp_var_name()

                # 1. 创建临时变量赋值语句: _p_temp_val_N = original_expression
                #    创建存储临时变量的 ast.Name 节点
                temp_name_store_node = ast.Name(id=temp_var_name, ctx=ast.Store())
                #    尝试从原始目标节点复制位置信息给临时变量名
                ast.copy_location(temp_name_store_node, node.targets[0])

                stmt1 = ast.Assign(
                    targets=[temp_name_store_node],
                    value=node.value  # 原始表达式
                )
                # 新语句的位置信息基于原始语句
                ast.copy_location(stmt1, node)


                # 2. 创建原始变量赋值语句: original_target = _p_temp_val_N
                #    创建加载临时变量的 ast.Name 节点
                temp_name_load_node = ast.Name(id=temp_var_name, ctx=ast.Load())
                #    位置信息也基于原始目标节点
                ast.copy_location(temp_name_load_node, node.targets[0])
                
                stmt2 = ast.Assign(
                    targets=node.targets,  # 原始赋值目标
                    value=temp_name_load_node
                )
                # 新语句的位置信息基于原始语句
                ast.copy_location(stmt2, node)
                # ast.unparse 会处理行号，通常 stmt2 的行号会比 stmt1 大。
                # 如果需要更精确控制，可以手动调整 stmt2.lineno = node.lineno + 1 (如果单行)
                # 但 ast.fix_missing_locations 通常能很好地处理。

                # 确保所有新创建的节点都有完整的源码位置信息
                # 这对于 ast.unparse 非常重要
                ast.fix_missing_locations(stmt1) # 递归填充 stmt1 及其子节点
                ast.fix_missing_locations(stmt2) # 递归填充 stmt2 及其子节点
                
                # 返回一个包含两个新语句的列表，替换原始的单个 Assign 节点
                return [stmt1, stmt2]
            else:
                # 对于复杂赋值（如元组解包）或非选定节点，不进行转换，
                # 并继续访问其子节点（如果适用）。
                return self.generic_visit(node)
        
        return self.generic_visit(node)


def apply_perturbation_assignment_temp_var(ast_root: ast.AST, threshold_ratio: float = 1.0) -> ast.AST:
    """
    应用"赋值语句 -> 临时变量赋值"扰动策略。

    Args:
        ast_root: 要转换的AST的根节点。
        threshold_ratio (float): 要扰动的已识别'Assign'语句的比例 (0.0 到 1.0)。
                                 1.0 表示全部扰动，0.5 表示扰动50%，等等。

    Returns:
        修改后的AST根节点。
    """
    # 1. 收集所有符合条件的 ast.Assign 候选节点
    candidate_assign_nodes = []
    for node in ast.walk(ast_root):
        if isinstance(node, ast.Assign):
            # 筛选条件：简单赋值 (targets 长度为 1)
            # 这样可以避免扰动像 a, b = 1, 2 这样的解包赋值，
            # 或者更复杂的赋值目标，这些可能需要更复杂的处理逻辑。
            if len(node.targets) == 1: # and isinstance(node.targets[0], (ast.Name, ast.Attribute, ast.Subscript)):
                candidate_assign_nodes.append(node)

    if not candidate_assign_nodes:
        return ast_root # 没有可扰动的赋值语句

    # 2. 根据 threshold_ratio 确定要扰动的节点数量
    num_to_perturb = int(len(candidate_assign_nodes) * threshold_ratio)
    
    # 3. 从候选中随机选择要实际转换的节点
    nodes_to_actually_transform = []
    if num_to_perturb > 0:
        # random.sample 的 k 参数必须非负且不大于总体数量
        actual_k = min(num_to_perturb, len(candidate_assign_nodes))
        if actual_k > 0 :
            nodes_to_actually_transform = random.sample(candidate_assign_nodes, actual_k)
    
    if not nodes_to_actually_transform:
        return ast_root # 没有选择任何节点进行扰动

    # 4. 创建并应用转换器
    # 注意：如果此函数可能在同一Python会话中多次调用，
    # 并且希望临时变量名在这些调用中是连续的，
    # 则不应在此处重置 AssignmentToTempVarTransformer._temp_var_counter。
    # 如果每次调用都希望从 _p_temp_val_1 开始，则应调用:
    # AssignmentToTempVarTransformer.reset_counter()
    # 目前框架每次调用 perturb_python_code 都会重新 parse，所以计数器会从新transformer实例的类变量开始
    
    transformer = AssignmentToTempVarTransformer(nodes_to_transform=nodes_to_actually_transform)
    modified_ast = transformer.visit(ast_root) 
    
    # 确保整个修改后的AST所有节点都有位置信息
    # visit 方法返回的是修改后的根节点
    return ast.fix_missing_locations(modified_ast) 

def count_perturbation_candidates(ast_root):
    count = 0
    for node in ast.walk(ast_root):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            count += 1
    return count 