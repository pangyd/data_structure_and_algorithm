# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2023/11/15 1:01
@desc: 
"""


class quick_sort():
    def partition(self, nums, left, right):
        l0 = nums[left]   # 空出第一个位置
        while left < right:
            while left < right and nums[right] >= l0:
                right -= 1
            nums[left] = nums[right]   # 比l0小的补到空出的第一个位置上
            while left < right and nums[left] <= l0:
                left += 1
            nums[right] = nums[left]   # 写入后left的位置为空
        nums[left] = l0
        return left

    def sort(self, nums, left, right):
        if left < right:
            ind = self.partition(nums, left, right)
            self.sort(nums, left, ind-1)
            self.sort(nums, ind+1, right)
        return nums
# qs = quick_sort()
# nums = [5, 5, 2, 4, 3, 8, 9, 3, 7]
# print(qs.sort(nums, 0, len(nums)-1))

def insert_sort(nums):
    for i in range(1, len(nums)):   # i表示摸到的牌
        j = i - 1
        l0 = nums[i]   # l0与前面全部值进行比较
        while j >= 0:   # 循环l0前面所有数
            if l0 <= nums[j]:
                nums[j+1] = nums[j]
            else:
                nums[j+1] = l0
                break
            j -= 1
        if j == -1:   # j中都比l0大
            nums[0] = l0
    return nums
# nums = [5, 1, 2, 4, 3, 8, 9, 6, 7]
# print(insert_sort(nums))


def merge_sort(nums):
    import math
    def fun(x):
        """x=123, return=0.123123123123..."""
        if x == 0:
            return 0
        L = int(math.log10(x)) + 1  # 位数
        return x / (10 ** L - 1)

    nums = [3, 30, 34, 5, 9, 260]
    nums.sort(key=fun, reverse=True)
    nums = list(map(str, nums))
    print(str(int(''.join(nums))))


class Top_k:
    def partition(self, nums, i, j):
        l0 = nums[i]
        # i, j = left, right
        while i < j:
            while i < j and nums[j] >= l0:
                j -= 1
            nums[i] = nums[j]
            while i < j and nums[i] <= l0:
                i += 1
            nums[j] = nums[i]
        nums[i] = l0
        return i


    def topk_ind(self, nums, k, left, right):
        if left < right:
            ind = self.partition(nums, left, right)
            if k == ind:
                return nums
            elif k > ind:
                self.topk_ind(nums, k, ind + 1, right)
            else:
                self.topk_ind(nums, k, left, ind - 1)
        return nums

    def topk_2(self, nums, k):
        import random
        def sort(nums, k):
            pivot = random.choice(nums)
            big, same, small = [], [], []
            for num in nums:
                if num == pivot:
                    same.append(num)
                elif num < pivot:
                    small.append(num)
                else:
                    big.append(num)
            if k <= len(big):
                return sort(big, k)
            elif k > len(big) + len(same):
                return sort(small, k-len(big)-len(same))
            else:
                return pivot
        res = sort(nums, k)
        return res


nums = [2, 1]
k = 1
left = 0
right = len(nums) - 1
topk = Top_k()
nums = topk.topk_ind(nums, len(nums) - k, left, right)
print(nums[len(nums) - k])