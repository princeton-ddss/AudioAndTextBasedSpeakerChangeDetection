from numpy import NaN, isnan


def map_notsure_to_false(x):
    return False if x == "NotSure" else x


def map_notsure_to_none(x):
    return None if x == "NotSure" else x


def aggregate_two_modes(x, y):
    return x if isnan(y) else True


def use_value_major(value_major, value_minor):
    """Always return value_major if it is not None."""
    if value_major != "NotSure":
        return value_major
    else:
        return value_minor


def map_string_to_bool(x):
    if isinstance(x, str):
        if x.lower() == "true":
            return True
        elif x.lower() == "false":
            return False
        else:
            raise NotImplementedError
    else:
        return x


def map_none_to_nan(x):
    return NaN if x is None else x
