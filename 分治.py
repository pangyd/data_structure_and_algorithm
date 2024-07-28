# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/4/9 9:44
@desc: 将问题的解拆分成多个子问题的解，再分别求子问题的解，最后将解合并
"""
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = None


"""按升序合并多个链表"""
def mergeTwoList(list1, list2):
    curNode = dummy = ListNode()
    while list1 and list2:
        if list1.val > list2.val:
            curNode.next = list2
            list2 = list2.next
        else:
            curNode.next = list1
            list1 = list1.next
        curNode = curNode.next
    curNode.next = list1 if list1 else list2
    return dummy.next


def mergeList(lists):
    if len(lists) == 0:
        return None
    if len(lists) == 1:
        return lists[0]
    mid = len(lists) // 2
    list1 = mergeList(lists[:mid])
    list2 = mergeList(lists[mid:])
    return mergeTwoList(list1, list2)

