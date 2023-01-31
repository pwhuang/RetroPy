# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from ufl.algebra import Abs
from ufl import min_value, max_value

class FluxLimiterCollection:
    """"""

    @staticmethod
    def minmod(r):
        return max_value(0.0, min_value(r, 1.0))

    @staticmethod
    def Osher(r, beta):
        return max_value(0.0, min_value(r, beta))

    @staticmethod
    def vanLeer(r):
        return (r + Abs(r))/(1.0 + Abs(r))

    @staticmethod
    def Koren(r):
        return max_value(0.0, min_value(2.0*r, min_value(1.0/3 + 2.0/3*r, 2.0)))

    @staticmethod
    def compact(r, alpha=0.4, beta=1.7, gamma=1.4):
        return max_value(0.0, min_value((2.0 + r)/3.0, max_value(min_value(min_value((2.0 + r)/3.0, gamma), beta*r), -alpha*r)))

    @staticmethod
    def MC(r):
        return max_value(0.0, min_value(min_value(2.0*r, 0.5*(1.0+r)), 2.0))
