"""Validate native time against integer controls with a float summation bound."""
import math
import sys


def validate(actual,control_steps,substeps):
    if type(control_steps) is not int or control_steps<0 or type(substeps) is not int or substeps not in (1,8):
        raise ValueError('Integer control clock and declared native subdivision required')
    expected=control_steps*.004
    # Floating-point addition error grows with both the number of additions
    # and clock magnitude. This remains far below one native tick throughout
    # the finite collection budget; integer counters detect missing ticks.
    bound=max(1e-9,2*sys.float_info.epsilon*control_steps*substeps*max(1.,expected))
    if not math.isfinite(actual) or abs(actual-expected)>bound:
        raise ValueError('World clock discontinuity')
    return dict(error_s=actual-expected,roundoff_bound_s=bound)
