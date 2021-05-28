from dolfin import SubDomain, near, DOLFIN_EPS

class MarkerCollection:
    """This class collects simple boundary marker instances."""

    class LeftBoundary(SubDomain):
        def __init__(self, xmin):
            super().__init__()
            self.xmin = xmin

        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], self.xmin, DOLFIN_EPS)

    class RightBoundary(SubDomain):
        def __init__(self, xmax):
            super().__init__()
            self.xmax = xmax

        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], self.xmax, DOLFIN_EPS)

    class BottomBoundary(SubDomain):
        def __init__(self, ymin):
            super().__init__()
            self.ymin = ymin

        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], self.ymin, DOLFIN_EPS)

    class TopBoundary(SubDomain):
        def __init__(self, ymax):
            super().__init__()
            self.ymax = ymax

        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], self.ymax, DOLFIN_EPS)

    class FrontBoundary(SubDomain):
        def __init__(self, zmin):
            super().__init__()
            self.zmin = zmin

        def inside(self, x, on_boundary):
            return on_boundary and near(x[2], self.zmin, DOLFIN_EPS)

    class BackBoundary(SubDomain):
        def __init__(self, zmax):
            super().__init__()
            self.zmax = zmax

        def inside(self, x, on_boundary):
            return on_boundary and near(x[2], self.zmax, DOLFIN_EPS)

    class AllBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
