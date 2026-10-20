"""Declared MFC setting grids, distinct from minimum operating flow."""
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math


def maximum_gas_setting(device):
    maximum = device.max_flow_sccm
    step = device.flow_rate_increment_sccm
    if maximum is None or step is None:
        return maximum
    upper = Decimal(str(maximum))
    spacing = Decimal(str(step))
    result = float((upper / spacing).to_integral_value(rounding=ROUND_FLOOR) * spacing)
    if result <= 0 or result < (device.min_flow_sccm or 0):
        raise ValueError("The declared MFC range contains no positive setting-grid point")
    return result


def gas_setting_at_least(device, requested):
    value = max(float(requested), float(device.min_flow_sccm or 0))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("MFC flow must be positive and finite")
    step = device.flow_rate_increment_sccm
    if step is not None:
        spacing = Decimal(str(step))
        count = (Decimal(str(value)) / spacing - Decimal("1e-9")).to_integral_value(rounding=ROUND_CEILING)
        value = float(count * spacing)
    maximum = maximum_gas_setting(device)
    if maximum is not None and value > maximum + 1e-10:
        raise ValueError("Required reagent-gas dose has no feasible MFC setting")
    return value


def gas_setting_supported(device, flow):
    step = device.flow_rate_increment_sccm
    return step is None or math.isclose(flow / step, round(flow / step), abs_tol=1e-7, rel_tol=0)
