import numpy as np


# Sphere Function
def sphere(x):
    return sum(x**2)


# implement pokutová funkce in cost based on omezení a vazby
def f1(x):
    assert len(x) == 5
    return np.sum(np.square(x))


def f2(x):
    assert len(x) == 5
    return 3*x[0] + 2*x[1] - x[2]/3 + 0.001*(x[3] - x[4])**3


def mo_cost(x, w: float):
    """Bi-objective cost function for 5 dimensional search space

    :param x: (Sequence[float]) input vector of length 5
    :param w: (float) weight scalar for weighted sum of objectives
    :return: (float) weighted sum of both objectives
    """
    assert len(x) == 5
    return w*f1(x) + (1.-w)*f2(x)


def g(x):
    return np.sum(np.square(x))


def rb(x, a: float):
    """ Rosenbrock function for boundary condition:
          g(x): x1**2 + x2**2 + x3**2 + x4**2 + x5**2 <= 10
    :param x: vector of input values
    :param a: wiggle criterium around the boundary
    """
    if g(x) > 10:
        r = 0
    elif g(x) > 10 - a:
        r = -1 + ((g(x) - 10 - a)/a)**2
    else:
        r = -1

    return r


def h1(x):  # == 0
    assert len(x) == 5
    return x[0] + 2*x[1] - x[2] - 0.5*x[3] + x[4] - 2


def h2(x):  # == 0
    assert len(x) == 5
    return 4*x[0] - 2*x[1] + 0.8*x[2] + 0.6*x[3] + 0.5*x[4]**2


def carroll(x, k: float):
    return (h1(x)**2 + h2(x)**2)/k


def cost_fun(x, w=0.5, a=0.1, k=0.01):
    """

    :param x: (Sequence[float]) input vector of length 5
    :param w: (float) weight scalar for weighted sum of objectives
    :param a: (float) wiggle criterium around the boundary for inequality boundary (Rosenbrock)
    :param k: (float) small positive value to scale carroll penalty function
    :return: value of aggregate cost function with respect to all boundaries (penalty functions)
    """
    mo = mo_cost(x, w)
    r = rb(x, a)
    c = carroll(x, k)
    # print(f"mo: {mo}, r: {r}, c: {c}")

    return mo*r + c


class Problem:
    costfunc = cost_fun
    nvar = 5
    varmin = -10.
    varmax = 10.


class Params:
    maxit = 1000
    npop = 256
    beta = 1.    # probability pressure for roulette wheel parent selection
    pc = 1.      # ratio of children from parents (if 1. there is same number of children as parents)
    gamma = 0.1  # (during crossover) possible overflow of alpha around 0 and 1 (allows better exploration)
    mu = 0.2     # mutation ratio
    sigma = 0.1  # mutation step size


class Individual:

    def __init__(self, position, costfunc, cost=None):
        self.position = self._ensure_bounds(position)
        self._costfunc = costfunc

        if cost:
            self.cost = cost
        else:
            self.cost = self._costfunc(position)

    @staticmethod
    def _ensure_bounds(position):
        position = np.maximum(position, Problem.varmin)
        position = np.minimum(position, Problem.varmax)
        return position

    def update(self, position):
        self.position = self._ensure_bounds(position)
        self.cost = self._costfunc(position)
