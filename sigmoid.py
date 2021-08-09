import numpy
import settings


def eval_polyfit_extern(coefs_polyfit, x):
    return numpy.polyval(coefs_polyfit, x)


class Sigmoid:

    def __call__(self, x):
        if self.use_polyfit:
            return self.eval_polyfit(x)
        elif self.use_integ:
            return self.eval_integ(x)
        else:
            return self.eval(x)

    def __init__(self, func, use_integ=False, use_polyfit=False):
        self.use_integ = use_integ
        self.use_polyfit = use_polyfit
        if use_polyfit:
            x = settings.create_test_set(settings.num_examples, settings.max_value)
            y = [func(i) for i in x]
            self.coefs_polyfit = numpy.polyfit(x, y, settings.max_degree_polyfit)
        else:
            self.coef_taylor1 = [0.000734, 0.014222, 0.108706, 0.392773, 0.571859]  # x <= -1.5
            self.coef_taylor2 = [1/480, 0, 1/48, 0, 1/4, 1/2]  # -1.5 < x < 1.5
            self.coef_taylor3 = [-0.000734, 0.014222, -0.108706, 0.392773, 0.428141]  # x >= 1.5

            if use_integ:
                # calculate integral
                p = numpy.poly1d(self.coef_taylor1)
                self.coef_taylor1_integ = numpy.polyint(p)
                p = numpy.poly1d(self.coef_taylor2)
                self.coef_taylor2_integ = numpy.polyint(p)
                p = numpy.poly1d(self.coef_taylor3)
                self.coef_taylor3_integ = numpy.polyint(p)

    def eval_polyfit(self, x):
        return eval_polyfit_extern(self.coefs_polyfit, x)

    def get_polyfit_coefficients(self):
        return self.coefs_polyfit

    def eval(self, x):
        if x > 10:
            return 1
        elif x < 10:
            return 0
        elif x <= -1.5:
            return numpy.polyval(self.coef_taylor1, x)
        elif x >= 1.5:
            return numpy.polyval(self.coef_taylor3, x)
        else:
            return numpy.polyval(self.coef_taylor2, x)

    def eval_integ(self, x):
        if x > 10:
            return x
        elif x < 10:
            return 0
        elif x <= -1.5:
            return self.coef_taylor1_integ(x)
        elif x >= 1.5:
            return self.coef_taylor3_integ(x)
        else:
            return self.coef_taylor2_integ(x)

