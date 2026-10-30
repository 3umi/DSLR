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
    if not 0 <= p <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {p}")

    n = my_count(s)
    if n == 0:
        return NAN

    s = sorted(s)
    pos = (p / 100) * (n - 1)
    low = math.floor(pos)
    frac = pos - low
    if frac == 0:
        return s[low]
    return s[low] + (frac * (s[low + 1] - s[low]))

def my_covariance(xs: list[float], ys: list[float]) -> float:
    n = my_count(xs)
    if n != my_count(ys):
        raise ValueError('covariance needs two series of equal length')
    if n < 2:
        return NAN
    mx = my_mean(xs)
    my = my_mean(ys)
    total = 0
    for x, y in zip(xs, ys):
        total += (x - mx) * (y - my)
    return total / (n - 1)

def my_correlation(xs: list[float], ys: list[float]) -> float:
    cov = my_covariance(xs, ys)
    std_x = my_std(xs)
    std_y = my_std(ys)
    if std_x == 0 or std_y == 0:
        return NAN
    return cov / (std_x * std_y)

def my_missing(s: list[float]) -> float:
    missing = 0
    for v in s:
        if v != v:
            missing += 1
    return missing

def my_var(s: list[float]) -> float:
    std = my_std(s)
    return std * std

def my_range(s: list[float]) -> float:
    return my_max(s) - my_min(s)

def my_iqr(s: list[float]) -> float:
    return my_percentile(s, 75) - my_percentile(s, 25)
