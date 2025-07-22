import ast

class NodeFinder(ast.NodeVisitor):
    """
    一个通用的访问者，用于查找AST中特定类型的节点。
    """
    def __init__(self, node_type_to_find):
        self.node_type_to_find = node_type_to_find
        self.found_nodes = []
        super().__init__()

    def visit(self, node):
        # hasattr is a simple way to check if it's an AST node we care about
        if hasattr(node, 'lineno') and isinstance(node, self.node_type_to_find):
            self.found_nodes.append(node)
        super().visit(node) # 继续访问子节点

    def find(self, ast_root):
        """对给定的AST树执行查找并返回找到的节点列表。"""
        self.found_nodes = []  # 重置以便于实例复用
        self.visit(ast_root)
        return self.found_nodes 