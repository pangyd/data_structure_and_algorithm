
def trap(self, height) -> int:
    """42.接雨水"""
    res = 0
    left, right = 0, len(height ) -1
    max_left, max_right = height[0], height[-1]
    while left < right:
        max_left = max(max_left, height[left])
        max_right = max(max_right, height[right])
        if height[left] < height[right]:
            res += max_left - height[left]
            left += 1
        else:
            res += max_right - height[right]
            right -= 1
    return res

def maxArea(height) -> int:
    """11.乘最多的水"""
    left, right = 0, len(height)-1
    res = 0
    while left < right:
        if height[left] > height[right]:
            res = max(res, height[right] * (right - left))
            right -= 1
        else:
            res = max(res, height[left] * (right - left))
            left += 1
    return res
