class TreeNode():
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None
def maxDepth(root) -> int:
    """104.二叉树最大深度"""
    if not root:
        return 0
    res = 0
    queue = [root]
    while queue:
        ll = []
        for node in queue:
            if node.left:
                ll.append(node.left.val)
            if node.right:
                ll.append(node.right.val)
        res += 1
        queue = ll
    return res

def invertTree(root):
    """226.翻转二叉树"""
    def bitree(root):
        if not root:
            return
        if root.left or root.right:
            tmp = root.left
            root.left = root.right
            root.right = tmp
        bitree(root.left)
        bitree(root.right)
    bitree(root)
    return root

def isSymmetric(root) -> bool:
    """101.判断是否为对称二叉树"""
    def pre_order(root):
        if not root:
            res1.append("")
        res1.append(root.val)
        pre_order(root.left)
        pre_order(root.right)
    def post_order(root):
        if not root:
            res2.append("")
        res2.append(root.val)
        post_order(root.left)
        post_order(root.right)
    res1, res2 = [], []
    pre_order(root)
    post_order(root)
    return res1 == res2[::-1]

def buildTree(preorder, inorder):
    """105.根据前序和中序构造二叉树"""
    if not preorder or not inorder:
        return
    root = TreeNode(preorder[0])
    ind = inorder.index(preorder[0])

    root.left = buildTree(preorder[1:ind+1], inorder[:ind])
    root.right = buildTree(preorder[ind+1:], inorder[ind+1:])
    return root

def flatten(root) -> None:
    """
    Do not return anything, modify root in-place instead.
    114.二叉树展开为列表
    """
    while root:
        if root.left:
            l = root.left
            while l.right:
                l = l.right
            l.right = root.right

            root.right = root.left
            root.left = None
        root = root.right

def hasPathSum(root, targetSum: int) -> bool:
    if not root:
        return False

    # 先序遍历
    def pre_order(root, init_val, k):
        if root:
            init_val += root.val
            if not root.left and not root.right and init_val == targetSum:  # 必须是叶子节点
                k = k + 1
            else:
                if not root.left and not root.right:
                    init_val -= root.val  # 遍历到叶子节点，次数累加！=目标值，则减去该叶子节点的值，继续遍历其他叶子节点
                k = pre_order(root.left, init_val, k)
                k = pre_order(root.right, init_val, k)
        return k

    k = pre_order(root, 0, 0)
    if k != 0:
        return True
    else:
        return False

def sumNumbers(root) -> int:
    def bitree(root, l):
        if not root:
            return
        if not root.left and not root.right:
            res.append(l+str(root.val))
        bitree(root.left, l+str(root.val))
        bitree(root.right, l+str(root.val))
    res = []
    bitree(root, "")
    res = [int(s) for s in res]
    return sum(res)

class TreeNode():
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None
def maxDepth(root) -> int:
    """104.二叉树最大深度"""
    if not root:
        return 0
    res = 0
    queue = [root]
    while queue:
        ll = []
        for node in queue:
            if node.left:
                ll.append(node.left.val)
            if node.right:
                ll.append(node.right.val)
        res += 1
        queue = ll
    return res

def invertTree(root):
    """226.翻转二叉树"""
    def bitree(root):
        if not root:
            return
        if root.left or root.right:
            tmp = root.left
            root.left = root.right
            root.right = tmp
        bitree(root.left)
        bitree(root.right)
    bitree(root)
    return root

def isSymmetric(root) -> bool:
    """101.判断是否为对称二叉树"""
    def pre_order(root):
        if not root:
            res1.append("")
        res1.append(root.val)
        pre_order(root.left)
        pre_order(root.right)
    def post_order(root):
        if not root:
            res2.append("")
        res2.append(root.val)
        post_order(root.left)
        post_order(root.right)
    res1, res2 = [], []
    pre_order(root)
    post_order(root)
    return res1 == res2[::-1]

def buildTree(preorder, inorder):
    """105.根据前序和中序构造二叉树"""
    if not preorder or not inorder:
        return
    root = TreeNode(preorder[0])
    ind = inorder.index(preorder[0])

    root.left = buildTree(preorder[1:ind+1], inorder[:ind])
    root.right = buildTree(preorder[ind+1:], inorder[ind+1:])
    return root

def flatten(root) -> None:
    """
    Do not return anything, modify root in-place instead.
    114.二叉树展开为列表
    """
    while root:
        if root.left:
            l = root.left
            while l.right:
                l = l.right
            l.right = root.right

            root.right = root.left
            root.left = None
        root = root.right

def hasPathSum(root, targetSum: int) -> bool:
    if not root:
        return False

    # 先序遍历
    def pre_order(root, init_val, k):
        if root:
            init_val += root.val
            if not root.left and not root.right and init_val == targetSum:  # 必须是叶子节点
                k = k + 1
            else:
                if not root.left and not root.right:
                    init_val -= root.val  # 遍历到叶子节点，次数累加！=目标值，则减去该叶子节点的值，继续遍历其他叶子节点
                k = pre_order(root.left, init_val, k)
                k = pre_order(root.right, init_val, k)
        return k

    k = pre_order(root, 0, 0)
    if k != 0:
        return True
    else:
        return False
