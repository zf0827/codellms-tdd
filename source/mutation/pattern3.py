# coding: utf-8
"""
模块名称: pattern3.py

本模块包含两种独立的扰动策略，旨在通过添加不直接贡献于原始算法计算的代码段来修改Python代码。
这些策略通过AST转换实现，并通过各自的 apply_perturbation_... 函数被外部框架（如 mutaor3.py）调用。

--- 扰动策略一: 打印日志 (Print Log) ---
模式识别: 单目标赋值语句。
    包括简单赋值 (例如: `x = expression`) 和带类型注解的赋值 (例如: `x: int = expression`)。
    仅当赋值操作有一个明确的目标时才会被考虑。

扰动策略: 在选定的单目标赋值语句之后，插入一条 `print` 语句，用于输出被赋值变量的名称及其值。
    原始形式:
        target_variable = some_expression
        # 或
        target_variable: type_hint = some_expression

    修改后形式:
        target_variable = some_expression
        print(f"DEBUG: target_variable_name_str = {target_variable_name_str}")
        # 或
        target_variable: type_hint = some_expression
        print(f"DEBUG: target_variable_name_str = {target_variable_name_str}")

使用方法:
    - 此策略主要通过 `apply_perturbation_print_log` 函数被扰动框架调用。
    - 调用参数:
        - `ast_root (ast.AST)`: 待扰动的 Python 代码的 AST 根节点。
        - `threshold_ratio (float)`: 一个介于 0.0 到 1.0 之间的浮点数，
          控制符合条件的赋值语句中被实际扰动的比例。
          例如，`threshold_ratio = 0.5` 表示大约一半符合条件的赋值语句后会添加日志打印。
    - 内部实现: 使用 `PrintLogTransformer` (一个 `ast.NodeTransformer` 子类) 来执行实际的 AST 转换。
    - 注意: `print` 语句的格式为 `f"DEBUG: {variable_name_as_string} = {variable_name_as_string}"`。
      获取 `variable_name_as_string` 对于复杂目标（如属性或下标访问）会尝试使用 `ast.unparse`。

--- 扰动策略二: 添加Try-Except包装并重抛 (Try-Except-Reraise Wrapper) ---
模式识别: 独立的函数调用语句。
    即那些作为表达式语句（`ast.Expr`）直接出现，并且其值为函数调用（`ast.Call`）的语句。
    例如: `my_function(arg1, arg2)`，而不是 `result = my_function() `中的调用。

扰动策略: 将选定的独立函数调用语句用一个 `try...except Exception: raise` 结构包装起来。
    此操作的目的是在不改变原始异常处理逻辑（因为异常被立即重抛）的前提下，
    在代码结构中引入 `try-except` 块。

    原始形式:
        module.some_function(param1, keyword_param=value)

    修改后形式:
        try:
            module.some_function(param1, keyword_param=value)
        except Exception:
            raise

使用方法:
    - 此策略主要通过 `apply_perturbation_try_except_reraise` 函数被扰动框架调用。
    - 调用参数:
        - `ast_root (ast.AST)`: 待扰动的 Python 代码的 AST 根节点。
        - `threshold_ratio (float)`: 一个介于 0.0 到 1.0 之间的浮点数，
          控制符合条件的独立函数调用语句中被实际扰动的比例。
          例如，`threshold_ratio = 1.0` 表示所有识别出的独立函数调用都将被包装。
    - 内部实现: 使用 `TryExceptReraiseWrapperTransformer` (一个 `ast.NodeTransformer` 子类)
      来执行实际的 AST 转换。
    - 效果: 如果被包装的函数调用在执行时抛出任何继承自 `Exception` 的异常，
      该异常会被捕获，然后立即被同一 `except` 块中的 `raise` 语句重新抛出，
      允许任何外部的异常处理机制按原始设计继续工作。
"""
import ast
import random

# --- 扰动策略：打印调试日志 ---

class PrintLogTransformer(ast.NodeTransformer):
    """
    AST转换器，在单目标赋值语句之后插入一个打印变量名和值的 print 语句。
    原始:
        x = value
    修改后:
        x = value
        print(f"DEBUG: x = {x}")
    """
    def __init__(self, nodes_to_transform):
        super().__init__()
        self.nodes_to_transform = set(nodes_to_transform)

    def _create_print_statement(self, target_node: ast.AST) -> ast.Expr | None:
        """
        为给定的赋值目标创建一个 print(f"DEBUG: name = {name}") 语句的AST节点。
        """
        if not isinstance(target_node, (ast.Name, ast.Attribute, ast.Subscript)):
            # 对于更复杂的赋值目标，直接 unparse 可能不理想或不安全
            # 为简化起见，我们主要关注 ast.Name，但尝试 unparse 其他简单情况
            try:
                target_name_str = ast.unparse(target_node)
            except Exception:
                return None #无法安全地获取名称字符串

        elif isinstance(target_node, ast.Name):
            target_name_str = target_node.id
        else: # Attribute, Subscript
             try:
                target_name_str = ast.unparse(target_node)
             except Exception:
                return None


        # 创建用于 print 的加载节点
        # 注意: target_node 本身通常是 Store ctx, 我们需要 Load ctx
        if isinstance(target_node, ast.Name):
            load_target_node = ast.Name(id=target_node.id, ctx=ast.Load())
        elif isinstance(target_node, ast.Attribute):
            load_target_node = ast.Attribute(value=target_node.value, attr=target_node.attr, ctx=ast.Load())
        elif isinstance(target_node, ast.Subscript):
            load_target_node = ast.Subscript(value=target_node.value, slice=target_node.slice, ctx=ast.Load())
        else:
            return None # 不支持的目标类型，或者无法安全创建加载节点

        ast.copy_location(load_target_node, target_node)

        # 构建 f-string: f"DEBUG: {target_name_str} = {load_target_node}"
        # 1. "DEBUG: "
        str_part_1 = ast.Constant(value=f"DEBUG: {target_name_str} = ")
        ast.copy_location(str_part_1, target_node)

        # 2. FormattedValue for the variable itself
        formatted_value = ast.FormattedValue(
            value=load_target_node,
            conversion=-1,  # no !s, !r, !a
            format_spec=None
        )
        ast.copy_location(formatted_value, target_node)

        joined_str = ast.JoinedStr(values=[str_part_1, formatted_value])
        ast.copy_location(joined_str, target_node)
        
        print_call = ast.Call(
            func=ast.Name(id='print', ctx=ast.Load()),
            args=[joined_str],
            keywords=[]
        )
        ast.copy_location(print_call, target_node)
        
        print_stmt = ast.Expr(value=print_call)
        ast.copy_location(print_stmt, target_node)
        
        return ast.fix_missing_locations(print_stmt)

    def visit_Assign(self, node: ast.Assign) -> list[ast.AST] | ast.AST:
        if node in self.nodes_to_transform and len(node.targets) == 1:
            # node.targets[0] 是赋值的目标
            print_stmt = self._create_print_statement(node.targets[0])
            if print_stmt:
                # 返回原始赋值语句和新的 print 语句
                # fix_missing_locations 确保新节点有位置信息
                return [node, ast.fix_missing_locations(print_stmt)]
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> list[ast.AST] | ast.AST:
        # AnnAssign: x: int = 10 (有值)
        if node in self.nodes_to_transform and node.value is not None:
            # node.target 是赋值的目标
            print_stmt = self._create_print_statement(node.target)
            if print_stmt:
                return [node, ast.fix_missing_locations(print_stmt)]
        return self.generic_visit(node)


def apply_perturbation_print_log(ast_root: ast.AST, threshold_ratio: float = 1.0) -> ast.AST:
    """
    应用"打印日志"扰动策略。
    在单目标赋值语句后插入打印该变量名和值的语句。
    """
    candidate_nodes = []
    for node in ast.walk(ast_root):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            candidate_nodes.append(node)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            candidate_nodes.append(node)

    if not candidate_nodes:
        return ast_root

    num_to_perturb = int(len(candidate_nodes) * threshold_ratio)
    nodes_to_actually_transform = []
    if num_to_perturb > 0:
        actual_k = min(num_to_perturb, len(candidate_nodes))
        if actual_k > 0:
            nodes_to_actually_transform = random.sample(candidate_nodes, actual_k)
    
    if not nodes_to_actually_transform:
        return ast_root

    transformer = PrintLogTransformer(nodes_to_transform=nodes_to_actually_transform)
    modified_ast = transformer.visit(ast_root)
    return ast.fix_missing_locations(modified_ast)


# --- 扰动策略：添加Try-Except包装并重抛 ---

class TryExceptReraiseWrapperTransformer(ast.NodeTransformer):
    """
    AST转换器，将独立的函数调用语句用 try...except Exception: raise 结构包装。
    原始:
        func_call(args)
    修改后:
        try:
            func_call(args)
        except Exception:
            raise
    """
    def __init__(self, nodes_to_transform):
        super().__init__()
        self.nodes_to_transform = set(nodes_to_transform)

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        # 目标是独立的函数调用，即 ast.Expr 其 value 是 ast.Call
        if node in self.nodes_to_transform and isinstance(node.value, ast.Call):
            original_call_expr_stmt = node # 这整个 ast.Expr 是 try 块的主体

            # 创建 except handler: except Exception: raise
            except_handler = ast.ExceptHandler(
                type=ast.Name(id='Exception', ctx=ast.Load()),
                name=None, # no 'as e'
                body=[ast.Raise()] #  Re-raise the current exception
            )
            ast.copy_location(except_handler, original_call_expr_stmt)
            ast.copy_location(except_handler.type, original_call_expr_stmt) # type node
            ast.copy_location(except_handler.body[0], original_call_expr_stmt) # raise node
            
            # 创建 try 节点
            try_node = ast.Try(
                body=[original_call_expr_stmt],
                handlers=[except_handler],
                orelse=[],
                finalbody=[]
            )
            ast.copy_location(try_node, original_call_expr_stmt)
            
            # 确保所有新创建的节点都有完整的源码位置信息
            return ast.fix_missing_locations(try_node)
            
        return self.generic_visit(node)


def apply_perturbation_try_except_reraise(ast_root: ast.AST, threshold_ratio: float = 1.0) -> ast.AST:
    """
    应用"Try-Except-Reraise包装"扰动策略。
    将独立的函数调用语句用 try...except Exception: raise 结构包装。
    """
    candidate_nodes = []
    for node in ast.walk(ast_root):
        # 寻找作为独立语句的函数调用
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            candidate_nodes.append(node)

    if not candidate_nodes:
        return ast_root

    num_to_perturb = int(len(candidate_nodes) * threshold_ratio)
    nodes_to_actually_transform = []
    if num_to_perturb > 0:
        actual_k = min(num_to_perturb, len(candidate_nodes))
        if actual_k > 0:
            nodes_to_actually_transform = random.sample(candidate_nodes, actual_k)
            
    if not nodes_to_actually_transform:
        return ast_root

    transformer = TryExceptReraiseWrapperTransformer(nodes_to_transform=nodes_to_actually_transform)
    modified_ast = transformer.visit(ast_root)
    return ast.fix_missing_locations(modified_ast) 
def count_perturbation_candidates_log(ast_root):
    count = 0
    for node in ast.walk(ast_root):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1) or (isinstance(node, ast.AnnAssign) and node.value is not None):
            count += 1
    return count

def count_perturbation_candidates_try(ast_root):
    count = 0
    for node in ast.walk(ast_root):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            count += 1
    return count