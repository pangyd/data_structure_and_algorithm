def convert(self, s: str, numRows: int) -> str:
    if len(s) == 1 or numRows == 1:
        return s
    res = [""] * numRows
    k = 0
    x = 2 * numRows - 2
    while k < len(s) // (2 * numRows - 2):
        i, j = 0, x - 1
        while i <= j:
            if i == 0 or i == numRows - 1:
                res[i] += s[i + k * x]
                i += 1
            else:
                res[i] += s[i + k * x] + s[j + k * x]
                i += 1
                j -= 1
        k += 1
    k = k * x

    if 0 < len(s) - k <= numRows:
        for i in range(len(s) - k):
            res[i] += s[i + k]
    elif len(s) - k > numRows:
        for i in range(numRows):
            res[i] += s[i + k]
        for i in range(len(s) - k - numRows):
            res[numRows - i - 2] += s[i + k + numRows]
    return "".join(res)