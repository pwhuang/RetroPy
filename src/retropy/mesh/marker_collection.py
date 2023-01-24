# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from numpy import isclose

class MarkerCollection:
    """This class collects simple boundary marker instances."""

    @staticmethod
    def LeftBoundary(x, xmin):
        return isclose(x[0], xmin)

    @staticmethod
    def RightBoundary(x, xmax):
        return isclose(x[0], xmax)

    @staticmethod
    def BottomBoundary(x, ymin):
        return isclose(x[1], ymin)

    @staticmethod
    def TopBoundary(x, ymax):
        return isclose(x[1], ymax)

    @staticmethod
    def FrontBoundary(x, zmin):
        return isclose(x[1], zmin)

    @staticmethod
    def BackBoundary(x, zmax):
        return isclose(x[1], zmax)

    @staticmethod
    def AllBoundary(x):
        return True

    class PeriodicBoundaryLeftRight:
        """
        This class makes the left/right boundaries of a rectangular mesh
        periodic.
        """

        pass