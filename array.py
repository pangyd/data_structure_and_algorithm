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