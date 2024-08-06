# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/3/21 11:40
@desc: 旋转排序数组
"""

def unrepeat_bool(nums, target):
    """找到nums中target的下标"""
    if len(nums) == 0:
        return False
    if len(nums) == 1:
        if nums[0] == target:
            return True
        return False
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return True   # return mid
        if nums[mid] >= nums[left]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return False


def repeat_bool(nums, target):
    if len(nums) == 0:
        return False
    if len(nums) == 1:
        if nums[0] == target:
            return True
        return False
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return True
        if nums[left] == nums[mid] == nums[right]:
            left += 1
            right -= 1
        elif nums[mid] >= nums[left]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return False


def unrepeat_value(nums):
    """找出最小值"""
    if len(nums) == 1:
        return nums[0]
    left = 0
    right = len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]


def repeat_value(nums):
    if len(nums) == 1:
        return nums[0]
    left = 0
    right = len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[left] == nums[mid] == nums[right]:
            left += 1
            right -= 1
        elif nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]


def repeat_value_2(nums):
    if len(nums) == 1:
        return nums[0]
    left = 0
    right = len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        elif nums[mid] < nums[right]:
            right = mid
        else:
            right -= 1
    return nums[left]