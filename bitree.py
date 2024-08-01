# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/4/2 11:48
@desc: 
"""

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
    114.二叉树展开为链表
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
    """是否存在头节点到根节点的总和等于targetSum"""
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
    """所有路线总和"""
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

def maxPathSum(root) -> int:
    """124.最大链路和：不一定要经过根节点"""
    def bitree(root):
        if not root:
            return 0
        l = bitree(root.left)   # 左子树最大链路和
        r = bitree(root.right)
        nonlocal res
        res = max(res, root.val+l+r)
        return max(max(l, r) + root.val, 0)   # 当前子树最大链路和
    res = float("-inf")
    bitree(root)
    return res

def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    """236.最近公共祖先"""
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    if left and right:
        return root
    elif left:
        return left
    elif right:
        return right

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
    res = [int(a) for a in res]
    return sum(res)

def getMinimumDifference(root) -> int:
    """530.二叉搜索树最小绝对值差"""
    def bitree(root, l):
        nonlocal res
        if not root:
            return
        l.append(root.val)
        if root.left and root.right:
            l_min = min([abs(a - root.left.val) for a in l])
            r_min = min([abs(a - root.right.val) for a in l])
            res = min(res, l_min, r_min)
        elif root.left and not root.right:
            l_min = min([abs(a - root.left.val) for a in l])
            res = min(res, l_min)
        elif not root.left and root.right:
            r_min = min([abs(a - root.right.val) for a in l])
            res = min(res, r_min)
        bitree(root.left, l)
        bitree(root.right, l)

    res = float("inf")
    bitree(root, [float("inf")])
    return res

class search_bitree:
    def demo(self, root):
        p = root
        st = []   # 栈
        while p or st:
            while p:
                st.append(p)
                p = p.left
            p = st.pop()
            proc(p.val)
            p = p.right
    def getMinimumDifference(self, root):
        """最小绝对值差"""
        st = []
        p = root
        pre = -float('inf')
        min_val = float('inf')
        while p is not None or st:
            while p is not None:
                st.append(p)
                p = p.left
            p = st.pop()
            cur = p.val
            if cur - pre < min_val:
                min_val = cur - pre
            pre = cur
            p = p.right
        return min_val

    def kthSmallest(self, root: TreeNode, k: int) -> int:
        st = []
        p = root
        s = 0
        while p is not None or st:
            while p is not None:
                st.append(p)
                p = p.left
            p = st.pop()
            s += 1
            if s == k:
                return p.val
            p = p.right
