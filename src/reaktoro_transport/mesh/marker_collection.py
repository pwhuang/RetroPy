from dolfin import SubDomain, near, DOLFIN_EPS
from numpy import arange, count_nonzero

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
        def __init__(self):
            super().__init__()

        def inside(self, x, on_boundary):
            return on_boundary

    class PeriodicBoundaryLeftRight(SubDomain):
        """
        This class makes the left/right boundaries of a rectangular mesh
        periodic.
        """

        def __init__(self, xmin, xmax, map_tol=1e-5):
            super().__init__(map_tol)
            self.xmin = xmin
            self.xmax = xmax

         # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool(x[0] < (self.xmin + DOLFIN_EPS) and\
                        x[0] > (self.xmin - DOLFIN_EPS) and\
                        on_boundary)

        # Map right boundary (H) to left boundary (G)
        def map(self, x, y):
            y[0] = x[0] - self.xmax
            y[1] = x[1]

        def mark(self, boundary_markers, periodic_markers, left_id, right_id):
            num_edges = len(boundary_markers.array())

            lb_mask = boundary_markers.array()==left_id
            rb_mask = boundary_markers.array()==right_id

            num_periodic_edges = self.__check_valid_periodic_boundary(lb_mask, rb_mask)

            left_idx = arange(num_edges, num_edges + num_periodic_edges*2, 2)
            right_idx = arange(num_edges+1, num_edges + num_periodic_edges*2, 2)

            periodic_markers.array()[lb_mask] = left_idx
            periodic_markers.array()[rb_mask] = right_idx

            return left_idx, right_idx

        def __check_valid_periodic_boundary(self, lb_mask, rb_mask):
            if count_nonzero(lb_mask)!=count_nonzero(rb_mask):
                raise Exception('number of periodic edges does not match!')

            return count_nonzero(lb_mask)
