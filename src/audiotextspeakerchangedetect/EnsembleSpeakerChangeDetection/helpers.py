from numpy import NaN, isnan

# Process Llama2 missing values
def map_notsure_to_false(x):
    if x == 'NotSure':
        return False
    else:
        return x

def map_notsure_to_none(x):
    if x == 'NotSure':
        return None
    else:
        return x

def aggregate_two_modes(x, y):
    if isnan(y): # if y is np.nan
        return x
    else:
        return True # False and True would return True


# Ensemble NLP Model Results with Others
def use_value_major(value_major, value_minor):
    # Always return value_major if it is not None
    if value_major != 'NotSure':
        return value_major
    else:
        return value_minor

def map_string_to_bool(x):
    if type(x) == str: # Check if x is string
        if x.lower() == 'true':
            return True
        elif x.lower() == 'false':
            return False
    else:
        return x

def map_none_to_nan(x):
    if x is None:  #shold not use not x because it would be correct when x is False or Noe or np.NaN
        return NaN
    else:
        return x
