"""Solve shared stoichiometric scales on declared pump setting grids."""

from fractions import Fraction
from math import ceil, floor, gcd, isfinite, isclose, lcm


def setting_supported(pump, flow):
    step = pump.flow_rate_increment_mL_min
    return step is None or isclose(flow / step, round(flow / step), abs_tol=1e-7, rel_tol=0)


def select_scale(weights, pumps, target, maximum=float("inf")):
    """Return nearest feasible scale and interval; never round feeds separately.

    Increments define a zero-origin setting grid. Missing increments retain
    continuous settings. Exact decimal ratios avoid float-derived false grids.
    """
    lower, upper = 0.0, maximum
    lattice = None
    for weight, pump in zip(weights, pumps):
        if pump is None:
            continue
        lower = max(lower, pump.min_flow_rate_mL_min / float(weight))
        upper = min(upper, pump.max_flow_rate_mL_min / float(weight))
        if pump.flow_rate_increment_mL_min is not None:
            period = Fraction(str(pump.flow_rate_increment_mL_min)) / weight
            lattice = period if lattice is None else Fraction(
                lcm(lattice.numerator, period.numerator),
                gcd(lattice.denominator, period.denominator),
            )
    if upper < lower - 1e-12:
        raise ValueError("No shared stoichiometric scale satisfies pump ranges and gas limits.")
    if lattice is None:
        return min(max(target, lower), upper), lower, upper, None
    step = float(lattice)
    first = max(1, ceil(lower / step - 1e-10))
    last = floor(upper / step + 1e-10) if isfinite(upper) else None
    if last is not None and first > last:
        raise ValueError("No shared stoichiometric scale satisfies the declared pump setting increments.")
    index = max(first, floor(target / step + 0.5))
    if last is not None:
        index = min(index, last)
    return float(index * lattice), lower, upper, step


def stream_weight(stream):
    return Fraction(str(stream.molar_equiv or 1.0)) / Fraction(str(stream.concentration_M))
