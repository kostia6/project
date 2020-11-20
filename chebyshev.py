import math
import settings
from sympy import *


def eval_extern(coefs, x):
    #print("eval_extern")
    a, b = settings.min_value, settings.max_value
    # assert(a <= x <= b)
    y = (2.0 * x - a - b) * (1.0 / (b - a))
    y2 = 2.0 * y
    (d, dd) = (coefs[-1], 0)
    for cj in coefs[-2:0:-1]:
        (d, dd) = (y2 * d - dd + cj, d)
    return y * d - dd + 0.5 * coefs[0]


class Chebyshev:

    def __call__(self, x):
        return self.eval(x)

    def __init__(self, a, b, n, func):
        self.a = a
        self.b = b
        self.func = func

        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)
        f = [func(math.cos(math.pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
        fac = 2.0 / n
        self.c = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / n)
                             for k in range(n)]) for j in range(n)]

    def eval(self, x):
        #print("eval")
        a, b = self.a, self.b
        #assert(a <= x <= b)
        y = (2.0 * x - a - b) * (1.0 / (b - a))
        y2 = 2.0 * y
        (d, dd) = (self.c[-1], 0)
        for cj in self.c[-2:0:-1]:
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * self.c[0]

    def get_coefficients(self):
        return self.c

    def print_str(self):
        difab = self.b - self.a
        sumab = self.a + self.b
        x = symbols('x')
        y_str = "(2.0 * x - (" + str(sumab) + "))*(1.0/(" + str(difab) + "))"
        #y_str = "(2.0 * x -(" + str(self.a) + ")-(" + str(self.b) + "))*(1.0/(" + str(self.b) + "-(" + str(self.a) + ")))"
        y2_str = "2.0 * (" + y_str + ")"

        y2_str_sim = simplify(y2_str)
        d = str(self.c[-1])
        dd = "0"
        for cj in self.c[-2:0:-1]:
            # d_sim = simplify(d).__str__()
            (d, dd) = ("(" + y2_str_sim.__str__() + " * (" + d + ")-(" + dd + ")+(" + str(cj)+")", d)

        return y_str + "*" + d + "-" + dd + "+0.5 *" + str(self.c[0])
