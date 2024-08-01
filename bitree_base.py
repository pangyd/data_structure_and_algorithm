# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/7/29 23:53
@desc: 
"""

class BinaryTree:
    def __init__(self, data):
        self.data = data
        self.lchild = None
        self.rchild = None
        self.parent = None

a = BinaryTree("A")
b = BinaryTree("B")
c = BinaryTree("C")
d = BinaryTree("D")
a.lchild = b
b.rchild = c
a.rchild = d

root = a

class BST():
    """二叉搜索树"""
    def __init__(self, data=None):
        self.root = None
        if data:
            for val in data:
                self.insert_no_recursion(val)
    def insert(self, node, val):   # 在node这棵树中插入val,插在叶子节点上
        """递归"""
        if not node:
            node = BinaryTree(val)
        elif val < node.data:
            node.lchild = self.insert(node.lchild, val)
            node.lchild.parent = node
        elif val > node.data:
            node.rchild = self.insert(node.rchild, val)
            node.rchild.parent = node
        return node
    def insert_no_recursion(self, val):
        p = self.root
        if not p:
            self.root = BinaryTree(val)
            return
        while True:   # !!!
            if val < p.data:
                if p.lchild:
                    p = p.lchild
                else:
                    p.lchild = BinaryTree(val)
                    p.lchild.parent = p
                    return
            elif val > p.data:
                if p.rchild:
                    p = p.rchild
                else:
                    p.rchild = BinaryTree(val)
                    p.rchild.parent = p
                    return
            else:
                return

    def query(self, node, val):
        if not node:
            return None
        if val > node.data:
            return self.query(node.rchild, val)
        elif val < node.data:
            return self.query(node.lchild, val)
        else:
            return node

    def query_np_recursion(self, val):
        p = self.root
        while p:
            if val > p.data:
                p = p.rchild
            elif val < p.data:
                p = p.lchild
            else:
                return p
        return None

    def __remove_node_1(self, node):
        """叶子节点"""
        if not node.parent:
            self.root = None
        if node == node.parent.lchild:
            node.parent.lchild = None
        if node == node.parent.rchild:
            node.parent.rchild = None

    def __remove_node_21(self, node):
        if not node.parent:
            self.root = node.lchild
            node.lchild.parent = None
        elif node == node.parent.lchild:
            node.parent.lchild = node.lchild
            node.lchild.parent = node.parent
        else:
            node.parant.rchild = node.rchile
            node.rchild.parent = node.parent

    def __remove_node_22(self, node):
        if not node.parent:
            self.root = node.rchild
            node.rchild.parent = None
        elif node == node.parent.lchild:
            node.parent.lchild = node.lchild
            node.lchild.parent = node.parent
        else:
            node.parent.rchild = node.rchild
            node.rchild.parent = node.parent

    def delete(self, val):
        if self.root:
            node = self.query_np_recursion(val)
            if not node:
                raise ValueError("not found")
            elif not node.lchild and not node.rchild:   # 叶子节点
                self.__remove_node_1(node)
            elif not node.rchild:   # 只有左子树
                self.__remove_node_21(node)
            elif not node.lchild:
                self.__remove_node_22(node)
            else:   # 有左右子树：找出右子树中最小节点
                min_node = node.rchild   # min_node一定没有左子树
                while min_node.lchild:
                    min_node = min_node.lchild
                node.data = min_node.data
                # 插入节点
                if min_node.rchild:
                    self.__remove_node_22(min_node)
                else:   # 叶子节点
                    self.__remove_node_1(min_node)

    def pre_order(self, root):
        """前序"""
        if root:
            print(root.data, end=",")
            self.pre_order(root.lchild)
            self.pre_order(root.rchild)

    def in_order(self, root):
        if root:
            self.in_order(root.lchild)
            print(root.data, end=",")
            self.in_order(root.rchild)

    def level_order(self, root):
        from collections import deque
        queue = deque()
        queue.append(root)
        while len(queue) > 0:
            node = queue.popleft()
            print(node.data, end=",")
            if node.lchild:
                queue.append(node.lchild)
            if node.rchild:
                queue.append(node.rchild)

# tree = BST([2, 5, 1, 7, 4, 3, 6])
# tree.in_order(tree.root)
# print("\n")
# tree.delete(2)
# tree.in_order(tree.root)