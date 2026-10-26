"""Chinese oral-style text normalization used before speech inference."""

import re

CN_DIGITS = "零一二三四五六七八九"


def _digit_by_digit(s: str) -> str:
    return "".join(CN_DIGITS[int(ch)] for ch in s)


def _four_to_cn(num: int) -> str:
    if num == 0:
        return ""
    units = ["", "十", "百", "千"]
    s = str(num)
    res, zero = [], False
    for i, ch in enumerate(s):
        d = int(ch)
        pos = len(s) - 1 - i
        if d == 0:
            zero = True
        else:
            if zero and res:
                res.append("零")
            zero = False
            res.append(CN_DIGITS[d] + units[pos])
    return "".join(res)


def _int_to_cn(n: int) -> str:
    if n == 0:
        return "零"
    if n < 0:
        return "负" + _int_to_cn(-n)

    big_units = ["", "万", "亿", "万亿"]
    groups = []
    while n > 0:
        groups.append(n % 10000)
        n //= 10000

    res, need_zero = [], False
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            if res:
                need_zero = True
            continue
        if res and (need_zero or g < 1000):
            if res[-1] != "零":
                res.append("零")
        res.append(_four_to_cn(g))
        res.append(big_units[i])
        need_zero = False

    result = "".join(res)
    if result.startswith("一十"):
        result = "十" + result[2:]
    return result


def _num_to_cn(s: str) -> str:
    if "." in s:
        a, b = s.split(".", 1)
        a_cn = _int_to_cn(int(a)) if a else "零"
        b_cn = "".join(CN_DIGITS[int(ch)] for ch in b)
        return a_cn + "点" + b_cn
    return _int_to_cn(int(s))


def _read_two_digit(mm: str) -> str:
    if mm == "00":
        return "零"
    if mm[0] == "0":
        return "零" + CN_DIGITS[int(mm[1])]
    return _int_to_cn(int(mm))


def _is_ascii_letter(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def _should_convert(s: str, start: int, end: int) -> bool:
    """Keep numbers adjacent to English letters in their original form."""
    i = end
    while i < len(s) and s[i] == " ":
        i += 1
    if i < len(s) and _is_ascii_letter(s[i]):
        return False

    j = start - 1
    while j >= 0 and s[j] == " ":
        j -= 1
    if j >= 0 and _is_ascii_letter(s[j]):
        return False

    return True


def _repl_time(m):
    h, mm, ss = m.group("h"), m.group("m"), m.group("s")
    h_cn = _int_to_cn(int(h))
    if ss is None:
        return h_cn + "点整" if mm == "00" else h_cn + "点" + _read_two_digit(mm)
    return f"{h_cn}点{_read_two_digit(mm)}分{_read_two_digit(ss)}秒"


def _repl_year(m):
    return _digit_by_digit(m.group("year_num")) + "年"


def _repl_month(m):
    return _int_to_cn(int(m.group("month_num"))) + "月"


def _repl_day(m):
    return _int_to_cn(int(m.group("day_num"))) + "日"


def _repl_percent(m):
    return "百分之" + _num_to_cn(m.group("pct"))


def _repl_decimal(m):
    return _num_to_cn(m.group("decimal"))


def _repl_int(m):
    return _num_to_cn(m.group("int"))


_PATTERN = re.compile(
    r"(?P<time>(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?)"
    r"|(?P<year>(?P<year_num>\d{2,4})年)"
    r"|(?P<month>(?P<month_num>\d{1,2})月)"
    r"|(?P<day>(?P<day_num>\d{1,2})日)"
    r"|(?P<percent>(?P<pct>\d+(?:\.\d+)?)%)"
    r"|(?P<decimal>\d+\.\d+)"
    r"|(?P<int>\d+)"
)


def _dispatch(m: re.Match) -> str:
    if not _should_convert(m.string, m.start(), m.end()):
        return m.group(0)

    kind = m.lastgroup
    if kind == "time":
        return _repl_time(m)
    if kind == "year":
        return _repl_year(m)
    if kind == "month":
        return _repl_month(m)
    if kind == "day":
        return _repl_day(m)
    if kind == "percent":
        return _repl_percent(m)
    if kind == "decimal":
        return _repl_decimal(m)
    if kind == "int":
        return _repl_int(m)
    return m.group(0)


def to_koubo_text(text: str) -> str:
    return _PATTERN.sub(_dispatch, text)
