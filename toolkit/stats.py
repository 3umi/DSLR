import math

NAN = float("nan")

def my_count(s: list[float]) -> float:
    count = 0
    for _ in s:
        count += 1
    return count

def my_mean(s: list[float]) -> float:
    n = my_count(s)
    if n == 0:
        return NAN
    sum = 0
    for v in s:
        sum += v
    return sum / n

def my_std(s: list[float]) -> float:
    n = my_count(s)
    if n < 2:
        return NAN

    dist = 0
    mean = my_mean(s)
    for v in s:
        dist += (v - mean) ** 2
    return math.sqrt((1 / (n - 1)) * dist)

def my_min(s: list[float]) -> float:
    if my_count(s) == 0:
        return NAN

    min = float('inf')
    for v in s:
        if v < min:
            min = v
    return min

def my_max(s: list[float]) -> float:
    if my_count(s) == 0:
        return NAN

    max = float('-inf')
    for v in s:
        if v > max:
            max = v
    return max

def my_percentile(s: list[float], p: float) -> float:
    n = my_count(s)
    if n == 0:
        return NAN
    if not 0 <= p <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {p}")
    
    pos = (p / 100) * (n - 1)
    low = math.floor(pos)
    frac = pos - low
    if frac == 0:
        return s[low]
    return s[low] + (frac * (s[low + 1] - s[low]))