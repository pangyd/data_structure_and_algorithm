# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/8/24 14:20
@desc: 
"""

def maxSubArray(nums) -> int:
    """53.最大子数组和"""
    res = tmp = nums[0]
    for i in range(1, len(nums)):
        if tmp + nums[i] > nums[i]:
            res = max(res, tmp + nums[i])
            tmp = nums[i] + tmp
        else:
            res = max(res, tmp, nums[i])
            tmp = nums[i]
    return res

def canCompleteCircuit(gas, cost) -> int:
    """134.加油站"""
    tmp = gas[0] - cost[0]
    res = [tmp, 0]
    for i in range(1, len(gas)):
        tmp = tmp + gas[i] - cost[i]
        if tmp <= res[0]:
            res = [tmp, i]
    if tmp < 0:
        return -1
    return (res[1]+1) % len(gas)


def maxSubarraySumCircular(nums) -> int:
    """918.环形子数组最大和"""
    max_s, min_s = float("-inf"), 0
    res1, res2 = 0, 0
    for num in nums:
        res1 = max(res1, 0) + num
        max_s = max(max_s, res1)
        res2 = min(res2, 0) + num
        min_s = min(min_s, res2)
    if sum(nums) == min_s:
        return max_s
    return max(max_s, sum(nums)-min_s)


def subarraySum(nums, k):
    """560.和为k的子数组"""
    dic = {0: 1}
    res = 0
    s = 0
    for num in nums:
        s += num
        tmp = s - k
        res += dic.get(tmp, 0) + 1
        dic[s] = dic.get(s, 0) + 1
    return res


def firstMissingPositive(nums) -> int:
    """41.缺失的第一个正数"""
    n = len(nums)
    hash_size = n + 1
    for i in range(n):
        if nums[i] <= 0 or nums[i] >= hash_size:
            nums[i] = 0
    for i in range(n):
        if nums[i] % hash_size != 0:
            pos = (nums[i] % hash_size) - 1
            nums[pos] = (nums[pos] % hash_size) + hash_size   # 说明pos+1位置的数存在
    for i in range(n):
        if nums[i] < hash_size:
            return i + 1
    return hash_size


def sortColors(nums) -> None:
    """
    Do not return anything, modify nums in-place instead.
    75.颜色分类
    """
    zero = 0
    two = len(nums) - 1
    i = 0
    while i <= two:
        if nums[i] == 0:
            tmp = nums[i]
            nums[i] = nums[zero]
            nums[zero] = tmp
            zero += 1
            i += 1
        elif nums[i] == 2:
            tmp = nums[i]
            nums[i] = nums[two]
            nums[two] = tmp
            two -= 1
        else:
            i += 1