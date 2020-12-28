from math import *
from decimal import Decimal

def p_root(value, p):
    root_value = 1 / float(p)
    return round(Decimal(value) ** Decimal(root_value), 3)


def minkowski_distance(x, y, p_value):
    return (p_root(sum(pow(abs(a - b), p_value)
                       for a, b in zip(x, y)), p_value))

