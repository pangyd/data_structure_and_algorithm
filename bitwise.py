# encoding: utf-8
"""
@author: PangYuda
@contact: px7939592023@163.com
@time: 2024/9/1 15:38
@desc: 
"""


def divide(self, dividend: int, divisor: int) -> int:
    """29.求除数"""
    if dividend == -2147483648 and divisor == -1:
        return 2147483647
    a, b, res = abs(dividend), abs(divisor), 0
    # 2 ** i * b <= a 《==》 a / b = 2 ** i + (a - 2 ** i * b) / b
    for i in range(31, -1, -1):
        if (b << i) <= a:
            res += 1 << i
            a -= b << i
    return res if (dividend > 0) == (divisor > 0) else -res