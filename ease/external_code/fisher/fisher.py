#! /usr/bin/env python

##############################################################################
# Following functions have been taken from the DendroPy library from:
##
## DendroPy Phylogenetic Computing Library.
##
## Copyright 2010 Jeet Sukumaran and Mark T. Holder.
## All rights reserved.
##
## See "LICENSE.txt" for terms and conditions of usage.
##
## If you use this work or any portion thereof in published work,
## please cite it as:
##
## Sukumaran, J. and M. T. Holder. 2010. DendroPy: a Python library
## for phylogenetic computing. Bioinformatics 26: 1569-1571.
##
##############################################################################

import math

## From dendropy.mathlib.probability


def hypergeometric_pmf(x, m, n, k):
    """
Given a population consisting of `m` items of class M and `n` items of class N,
this returns the probability of observing `x` items of class M when sampling
`k` times without replacement from the entire population (i.e., {M,N})

p(x) = (choose(m, x) * choose(n, k-x)) / choose(m+n, k)
"""
    # following fails with 'OverflowError: long int too large to convert to
    # float' with large numbers
    # return float(binomial_coefficient(m, x) * binomial_coefficient(n, k-x))/binomial_coefficient(m+n, k)
    a = math.log(binomial_coefficient(m, x))
    b = math.log(binomial_coefficient(n, k -x))
    c = math.log(binomial_coefficient(m +n, k))
    return math.exp(a +b -c)

## From dendropy.mathlib.probability


def binomial_coefficient(population, sample):
    "Returns `population` choose `sample`."
    s = max(sample, population - sample)
    assert s <= population
    assert population > -1
    if s == population:
        return 1
    numerator = 1
    denominator = 1
    for i in xrange(s +1, population + 1):
        numerator *= i
        denominator *= (i - s)
    return numerator /denominator

## From dendropy.mathlib.statistics


class FishersExactTest(object):
    """
Given a 2x2 table:

+---+---+
| a | b |
+---+---+
| c | d |
+---+---+

represented by a list of lists::

[[a,b],[c,d]]

this calculates the sum of the probability of this table and all others
more extreme under the null hypothesis that there is no association between
the categories represented by the vertical and horizontal axes.
"""

    def probability_of_table(table):
        """
Given a 2x2 table:

+---+---+
| a | b |
+---+---+
| c | d |
+---+---+

represented by a list of lists::

[[a,b],[c,d]]

this returns the probability of this table under the null hypothesis of
no association between rows and columns, which was shown by Fisher to be
a hypergeometric distribution:

p = ( choose(a+b, a) * choose(c+d, c) ) / choose(a+b+c+d, a+c)

"""
        a = table[0][0]
        b = table[0][1]
        c = table[1][0]
        d = table[1][1]
        return hypergeometric_pmf(a, a +b, c +d, a +c)
    probability_of_table = staticmethod(probability_of_table)

    def __init__(self, table):
        self.table = table
        self.flat_table = [table[0][0], table[0][1], table[1][0], table[1][1]]
        self.min_value = min(self.flat_table)
        self.max_value = max(self.flat_table)

    def _rotate_cw(self, table):
        """
Returns a copy of table such that all the values
are rotated clockwise once.
"""
        return [ [ table[1][0], table[0][0] ],
                [table[1][1], table[0][1] ] ]

    def _min_rotation(self):
        """
Returns copy of self.table such that the smallest value is in the first
(upper left) cell.
"""
        table = [list(self.table[0]), list(self.table[1])]
        while table[0][0] != self.min_value:
            table = self._rotate_cw(table)
        return table

    def _max_rotation(self):
        """
Returns copy of self.table such that the largest value is in the first
(upper left) cell.
"""
        table = [list(self.table[0]), list(self.table[1])]
        while table[0][0] != self.max_value:
            table = self._rotate_cw(table)
        return table

    def _sum_left_tail(self):
        # left_tail_tables = self._get_left_tail_tables()
        # p_vals = [ self.probability_of_table(t) for t in left_tail_tables ]
        p_vals = self._get_left_tail_probs()
        return sum(p_vals)

    def _sum_right_tail(self):
        # right_tail_tables = self._get_right_tail_tables()
        # p_vals = [ self.probability_of_table(t) for t in right_tail_tables ]
        p_vals = self._get_right_tail_probs()
        return sum(p_vals)

    def _get_left_tail_probs(self):
        table = self._min_rotation()
        row_totals = [sum(table[0]), sum(table[1])]
        col_totals = [table[0][0] + table[1][0], table[0][1] + table[1][1]]
        p_vals = []
        while True:
            table[0][0] -= 1
            if table[0][0] < 0:
                break
            table[0][1] = row_totals[0] - table[0][0]
            table[1][0] = col_totals[0] - table[0][0]
            table[1][1] = row_totals[1] - table[1][0]
            p_vals.append(self.probability_of_table(table))
        return p_vals

    def _get_right_tail_probs(self):
        table = self._min_rotation()
        row_totals = [sum(table[0]), sum(table[1])]
        col_totals = [table[0][0] + table[1][0], table[0][1] + table[1][1]]
        p_vals = []
        while True:
            table[0][0] += 1
            table[0][1] = row_totals[0] - table[0][0]
            if table[0][1] < 0:
                break
            table[1][0] = col_totals[0] - table[0][0]
            if table[1][0] < 0:
                break
            table[1][1] = row_totals[1] - table[1][0]
            if table[1][1] < 0:
                break
            p_vals.append(self.probability_of_table(table))
        return p_vals

    def _get_left_tail_tables(self):
        table = self._min_rotation()
        row_totals = [sum(table[0]), sum(table[1])]
        col_totals = [table[0][0] + table[1][0], table[0][1] + table[1][1]]
        left_tail_tables = []
        while True:
            table[0][0] -= 1
            if table[0][0] < 0:
                break
            table[0][1] = row_totals[0] - table[0][0]
            table[1][0] = col_totals[0] - table[0][0]
            table[1][1] = row_totals[1] - table[1][0]
            left_tail_tables.append([list(table[0]), list(table[1])])
        return left_tail_tables

    def _get_right_tail_tables(self):
        table = self._min_rotation()
        row_totals = [sum(table[0]), sum(table[1])]
        col_totals = [table[0][0] + table[1][0], table[0][1] + table[1][1]]
        right_tail_tables = []
        while True:
            table[0][0] += 1
            table[0][1] = row_totals[0] - table[0][0]
            if table[0][1] < 0:
                break
            table[1][0] = col_totals[0] - table[0][0]
            if table[1][0] < 0:
                break
            table[1][1] = row_totals[1] - table[1][0]
            if table[1][1] < 0:
                break
            right_tail_tables.append([list(table[0]), list(table[1])])
        return right_tail_tables

    def left_tail_p(self):
        """
Returns the sum of probabilities of this table and all others more
extreme.
"""
        return self.probability_of_table(self.table) + self._sum_left_tail()

    def right_tail_p(self):
        """
Returns the sum of probabilities of this table and all others more
extreme.
"""
        return self.probability_of_table(self.table) + self._sum_right_tail()

    def two_tail_p(self):
        """
Returns the sum of probabilities of this table and all others more
extreme.
"""
        p0 = self.probability_of_table(self.table)
        all_p_vals = self._get_left_tail_probs() + self._get_right_tail_probs()
        p_vals = []
        for p in all_p_vals:
            if p <= p0:
                p_vals.append(p)
        return sum(p_vals) + p0


def assert_almost_equal(v1, v2, prec=8):
    if abs(v1 -v2) <= 10 **(-prec):
        print "OK: {} == {}".format(v1, v2)
    else:
        print "FAIL: {} != {}".format(v1, v2)

if __name__ == "__main__":
    table = [[12, 5], [29, 2]]
    ft = FishersExactTest(table)
    assert_almost_equal(ft.left_tail_p(), 0.044554737835078267)
    assert_almost_equal(ft.right_tail_p(), 0.99452520602190897)
    assert_almost_equal(ft.two_tail_p(), 0.08026855207410688)
