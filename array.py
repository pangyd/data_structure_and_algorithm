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
