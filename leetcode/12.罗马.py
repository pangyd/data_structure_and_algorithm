def intToRoman(num: int) -> str:
    # 使用哈希表，按照从大到小顺序排列
    hashmap = {1000: 'M', 900: 'CM', 500: 'D', 400: 'CD', 100: 'C', 90: 'XC', 50: 'L', 40: 'XL', 10: 'X', 9: 'IX',
               5: 'V', 4: 'IV', 1: 'I'}
    res = ''
    for key in hashmap:
        if num // key != 0:
            count = num // key
            res += hashmap[key] * count
            num %= key
    return res


def romanToInt(s: str) -> int:
    match_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    if len(s) == 1:
        return match_dict[s[0]]
    res = 0
    i = 0
    n = len(s)
    while i < n-1:
        if s[i] == "I" and s[i+1] in ["V", "X"]:
                res += match_dict[s[i+1]] - match_dict[s[i]]
                i += 2
        elif s[i] == "X" and s[i+1] in ["L", "C"]:
                res += match_dict[s[i+1]] - match_dict[s[i]]
                i += 2
        elif s[i] == "C" and s[i+1] in ["D", "M"]:
                res += match_dict[s[i+1]] - match_dict[s[i]]
                i += 2
        else:
            res += match_dict[s[i]]
            i += 1
    if i == n:
        return res
    if i == n - 1:
        res += match_dict[s[-1]]
        return res