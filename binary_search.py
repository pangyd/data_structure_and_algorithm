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
        if nums[mid] >= nums[left]:   # 判断mid位置的大小
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

def findPeakElement(nums) -> int:
    """162.寻找峰值 -- 任意一个"""
    if len(nums) == 1:
        return 0
    elif len(nums) == 2:
        return nums.index(max(nums))
    left, right = 0, len(nums)
    while left <= right:
        mid = left + (right - left) // 2
        if 0 < mid < len(nums) - 1:
            if nums[mid-1] < nums[mid] and nums[mid+1] < nums[mid]:
                return mid
            elif nums[mid-1] >= nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        elif mid == 0:
            if nums[mid] > nums[mid+1]:
                return mid
            else:
                left = mid + 1
        else:
            if nums[mid] > nums[mid-1]:
                return mid
            else:
                right = mid - 1

def searchRange(nums, target: int):
    """34.排序数组中查找第一个和最后一个元素"""
    if not nums:
        return [-1, -1]
    res = [-1, -1]
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            res = [mid, mid]
            mid1, mid2 = mid-1, mid+1
            while mid1 >= 0 and nums[mid1] == target:
                res[0] = mid1
                mid1 -= 1
            while mid2 < len(nums) and nums[mid2] == target:
                res[1] = mid2
                mid2 += 1
            return res
        elif nums[mid] > target:
            right = mid
        else:
            left = mid + 1
    return res

def findMedianSortedArrays(nums1, nums2) -> float:
    """4.寻找两个正序数组的中位数"""
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    n1, n2 = len(nums1), len(nums2)

    left, right, half_len = 0, n1, (n1 + n2 + 1) // 2
    mid1 = (left + right) // 2
    mid2 = half_len - mid1

    while left < right:
        if left < n1 and nums1[mid1] < nums2[mid2 - 1]:
            left = mid1 + 1
        else:
            right = mid1
        mid1 = (left + right) // 2
        mid2 = half_len - mid1

    if mid1 == 0:
        max_left = nums2[mid2 - 1]
    elif mid2 == 0:
        max_left = nums1[mid1 - 1]
    else:
        max_left = max(nums1[mid1 - 1], nums2[mid2 - 1])

    if (n1 + n2) % 2 == 1:
        return max_left

    if mid1 == n1:
        max_right = nums2[mid2]
    elif mid2 == n2:
        max_right = nums1[mid1]
    else:
        max_right = min(nums1[mid1], nums2[mid2])

    return (max_left + max_right) / 2