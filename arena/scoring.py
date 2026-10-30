"""Documented placement scoring; no poker engine reimplementation."""
def points(values):
    """Descending N..1 points, tied competitors share occupied places."""
    return [sum(len(values) - j for j, value in enumerate(sorted(values, reverse=True)) if value == x) / values.count(x) for x in values]


def table_sizes(count):
    if count < 2:
        raise ValueError('At least two agents are required')
    if count <= 6:
        return [count]
    if count == 7:
        return [4, 3]
    candidates = [k for k in range(1, count + 1) if 4 * k <= count <= 6 * k]
    groups = min(candidates, key=lambda k: abs(count / k - 5))
    small, extra = divmod(count, groups)
    return [small + (i < extra) for i in range(groups)]
