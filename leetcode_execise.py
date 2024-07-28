# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/3/27 16:36
@desc: 
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

a = ListNode()
a.next = ListNode(1)
b = a   # 当前节点为1
a.next = ListNode(2)


# from torch.optim import lr_scheduler, SGD
# optimizer = SGD()
# scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
# scheduler.step()

# num = 5
# print(type(bin(num)))
# ans = sum([int(x) for x in list(str(bin(num)))])
# print(ans)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def rec_tree(left, right):
    all_trees = []
    if left > right:
        return [None]
    for i in range(left, right + 1):
        left_tree = rec_tree(left, i - 1)
        right_tree = rec_tree(i + 1, right)
        for l in left_tree:
            for r in right_tree:
                cur_node = TreeNode(i)
                cur_node.left = l
                cur_node.right = r
                all_trees.append(cur_node)
    return all_trees


# res = rec_tree(1, 4)

a = TreeNode(1)
b = TreeNode(3)
c = TreeNode(2)
root = a
a.left = b
b.right = c


a = [1, 2, 3, 4, 5]
a.append(None)
print(a[::-1])