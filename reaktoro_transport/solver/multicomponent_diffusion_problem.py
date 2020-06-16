#from . import np, rkt
from . import *

class multicomponent_diffusion_problem(reactive_transport_problem_base):
    def __init__(self):
        super().__init__()
        self.mass_bal_violation = 0

    def set_transport_species(self, num_transport_components, initial_expr):
        self.num_transport_components = num_transport_components

        self.function_space = dolfin.FunctionSpace(self.mesh, 'CG', 1)

        self.X_list = []      # Xn
        self.X_list_old = []  # Xn-1
        self.mu_list = []
        #self.grad_psi = dolfin.Function(self.function_space)

        for i in range(self.num_transport_components):
            self.X_list.append(dolfin.Function(self.function_space))
            self.X_list_old.append(dolfin.interpolate(initial_expr[i], self.function_space))

        for i in range(self.num_components):
            self.mu_list.append(dolfin.Function(self.function_space))

        self.num_dof = len(self.X_list[0].vector())

        return self.X_list_old

    def set_boundary_conditions(self):
        # The user should override this function to define boundary conditions!

        self.bc_list = \
        [dolfin.DirichletBC(self.function_space, dolfin.Constant(0.0), self.boundary_markers, 2),
         dolfin.DirichletBC(self.function_space, dolfin.Constant(0.0), self.boundary_markers, 1)]

    def set_transport_equations(self):
        v = dolfin.TestFunction(self.function_space)
        u = dolfin.TrialFunction(self.function_space)

        #self.u_n   = dolfin.Function(self.function_space)
        #self.mu    = dolfin.Function(self.function_space)
        R = 8.314 # Ideal gas constant

        ds = dolfin.Measure('ds', domain=self.mesh, subdomain_data=self.boundary_markers)
        dx = dolfin.Measure('dx', domain=self.mesh, subdomain_data=self.boundary_markers)

        # Placeholder for dt
        self.dt = dolfin.Constant(1.0)
        self.solver_list = []

        # The form of grad_psi/F, assuming the solvent has no charge, e.g. H2O(l).
        grad_psi = dolfin.Constant(0.0)*grad(self.mu_list[0])
        denom = dolfin.Constant(0.0)
        for i in range(self.num_transport_components):
            grad_psi += Constant(self.D_list[i]*self.z_list[i]/self.M_list[i])\
                         *self.X_list_old[i]*grad(self.mu_list[i])

            denom += Constant(self.D_list[i]*self.z_list[i]**2/self.M_list[i])*self.X_list_old[i]

        grad_psi = -grad_psi/denom

        a = v*u/self.dt*dx
        for i in range(self.num_transport_components):
            L = v*self.X_list_old[i]/self.dt*dx\
                - Constant(self.D_list[i]/(R*self.T))\
                 *self.X_list_old[i]\
                 *inner(grad(self.mu_list[i]) + Constant(self.z_list[i])*grad_psi, grad(v))*dx\

            linear_problem = dolfin.LinearVariationalProblem(a, L, self.X_list[i], bcs=self.bc_list)
            self.solver_list.append(dolfin.LinearVariationalSolver(linear_problem))

    def solve_chemical_equilibrium(self):
        # Another method that needs to be overridden

        for i in range(self.num_dof):
            solvent_mass = 1.0 # Initialize solvent mass fraction
            for j in range(self.num_transport_components):
                if self.X_list[j].vector()[i] < 0.0:
                    self.mass_bal_violation += 1
                    self.chem_state.setSpeciesMass(j, 1e-16, 'kg')
                else:
                    solvent_mass -= self.X_list[j].vector()[i]
                    self.chem_state.setSpeciesMass(j, self.X_list[j].vector()[i], 'kg')

            self.chem_state.setSpeciesMass(self.num_components-1, solvent_mass, 'kg')
            self.chem_equi_solver.solve(self.chem_state)

            chem_prop = self.chem_state.properties()

            for j in range(self.num_transport_components):
                self.mu_list[j].vector()[i] = chem_prop.chemicalPotentials().val[j]
                self.X_list[j].vector()[i] = self.chem_state.speciesAmount(j, 'mol')*self.M_list[j]

            del chem_prop

    def solve(self, dt_num, timesteps):
        out_list = []
        for i in range(self.num_transport_components):
            out_list.append([self.X_list_old[i].copy()])

        self.dt.assign(dolfin.Constant(dt_num))

        for i in range(timesteps):
            self.solve_chemical_equilibrium()
            for j in range(self.num_transport_components):
                self.solver_list[j].solve()

                self.X_list_old[j].assign(self.X_list[j])

                out_list[j].append(self.X_list_old[j].copy())

        return out_list
