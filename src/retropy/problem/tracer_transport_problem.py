# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *

class TracerTransportProblem(TransportProblemBase,
                             MassBalanceBase, ComponentProperty):
    """A class that solves single-phase tracer transport problems."""

    def __init__(self, mesh, boundary_markers, interior_markers,
                 domain_markers, marker_dict, facet_dict,
                 option='cell_centered', periodic_bcs=None):
        try:
            super().num_forms
        except:
            raise Exception("num_forms does not exist. Consider inherit a solver class.")

        self.set_mesh(mesh, option, periodic_bcs)
        self.set_boundary_markers(boundary_markers)
        self.set_interior_markers(interior_markers)
        self.set_domain_markers(domain_markers)
        self.marker_dict = marker_dict
        self.facet_dict = facet_dict

        self.__dirichlet_bcs = []

        # TODO: Where should time be defined? There may be a better way to manage time.
        self.current_time = Constant(self.mesh, 0.0)
        self.dt = Constant(self.mesh, 1.0)

    def mark_component_boundary(self, **kwargs):
        """This method gives boundary markers physical meaning.

        Keywords
        ---------
        {component_name : markers}
        Example: {'Na+': [1, 2, 3], 'outlet': [4]}
        """

        self.__boundary_dict = kwargs

    def set_component_fe_space(self):
        self.FiniteElement = FiniteElement(super().fe_space,
                                           self.mesh.ufl_cell(),
                                           super().fe_degree)

        element_list = [self.FiniteElement for _ in range(self.num_component)]

        self.comp_func_spaces = FunctionSpace(self.mesh, MixedElement(element_list))

        self.fluid_components = Function(self.comp_func_spaces)

        if self.num_component > 1:
            self.fluid_comp_sub = self.fluid_components.split()
        else:
            self.fluid_comp_sub = [self.fluid_components.sub(0)]

        self.periodic_flux = Function(self.comp_func_spaces)

        self.func_space_list = []

        for name, idx in self.component_dict.items():
            self.func_space_list.append(self.comp_func_spaces.sub(idx).collapse()[0])
            self.fluid_comp_sub[idx].name = name

    def get_num_dof_per_component(self):
        return len(self.fluid_components.vector[:].reshape(-1, self.num_component))

    def get_function_space(self):
        return self.comp_func_spaces

    def get_fluid_components(self):
        """Get the fluid components as ufl vector."""

        return as_vector([self.fluid_comp_sub[i] for i in range(self.num_component)])

    def initialize_form(self):
        """"""

        self.__u = TrialFunction(self.comp_func_spaces)
        self.__w = TestFunction(self.comp_func_spaces)

        self.tracer_forms = [Constant(self.mesh, 0.0)*inner(self.__w, self.__u)*self.dx]*super().num_forms

    def get_trial_function(self):
        return self.__u

    def set_advection_velocity(self):
        self.advection_velocity = \
        as_vector([self.fluid_velocity for _ in range(self.num_component)])

    def set_component_ics(self, name, expressions):
        idx = self.component_dict[name]
        self.fluid_components.sub(idx).interpolate(expressions)

    def add_component_advection_bc(self, component_name: str, values, kappa: Any = 1, f_id = 0):
        """"""

        if len(values)!=len(self.__boundary_dict[component_name]):
            raise Exception("length of values != number of markers")

        idx = self.component_dict[component_name]
        markers = self.__boundary_dict[component_name]

        for i, marker in enumerate(markers):
            self.tracer_forms[f_id] += kappa*self.advection_flux_bc(self.__w[idx], values[i], marker)

    def add_component_flux_bc(self, component_name, values, kappa: Any = 1, f_id = 0):
        """"""

        if len(values)!=len(self.__boundary_dict[component_name]):
            raise Exception("length of values != number of markers")

        idx = self.component_dict[component_name]
        markers = self.__boundary_dict[component_name]

        for i, marker in enumerate(markers):
            self.tracer_forms[f_id] += kappa*self.general_flux_bc(self.__w[idx],
                                                                  values[i], marker)

    def add_component_diffusion_bc(self, component_name, diffusivity, values, kappa: Any = 1, f_id = 0):
        """"""

        if len(values)!=len(self.__boundary_dict[component_name]):
            raise Exception("length of values != number of markers")

        idx = self.component_dict[component_name]
        markers = self.__boundary_dict[component_name]

        for i, marker in enumerate(markers):
            self.tracer_forms[f_id] += kappa*self.diffusion_flux_bc(self.__w[idx], self.__u[idx],
                                                                    diffusivity, values[i], marker)

    def add_component_dirichlet_bc(self, component_name, values):
        """"""

        if len(values)!=len(self.__boundary_dict[component_name]):
            raise Exception("length of values != number of markers")

        idx = self.component_dict[component_name]

        # I have no idea how this works. Need to read more fenicsx documentation to find out.
        for i, facet in enumerate(self.facet_dict.values()):
            dof = locate_dofs_topological(self.comp_func_spaces, self.mesh.topology.dim - 1, facet)
            bc = dirichletbc(values[i], dof, self.comp_func_spaces.sub(idx))
            self.__dirichlet_bcs.append(bc)

    def add_outflow_bc(self, f_id=0):
        """"""

        for i, marker in enumerate(self.__boundary_dict['outlet']):
            self.tracer_forms[f_id] += self.advection_outflow_bc(self.__w, self.__u, marker)

    def add_time_derivatives(self, u, kappa: Any = 1, f_id = 0):
        self.tracer_forms[f_id] += kappa*self.d_dt(self.__w, self.__u, u)

    def add_explicit_advection(self, u, kappa: Any = 1, marker = 0, f_id = 0):
        """Adds explicit advection physics to the variational form."""

        self.tracer_forms[f_id] += kappa*self.advection(self.__w, u, marker)

    def add_explicit_downwind_advection(self, u, kappa: Any = 1, marker = 0, f_id = 0):
        self.tracer_forms[f_id] += kappa*self.downwind_advection(self.__w, u, marker)

    def add_explicit_centered_advection(self, u, kappa: Any = 1, marker = 0, f_id = 0):
        self.tracer_forms[f_id] += kappa*self.centered_advection(self.__w, u, marker)

    def add_explicit_periodic_advection(self, u, kappa: Any = 1, marker_l = [], marker_r = [], f_id=0):
        # This method has not yet been tested.
        # TODO: Write a test and test it.

        for ml, mr in zip(marker_l, marker_r):
            self.tracer_forms[f_id] += kappa*self.periodic_advection(self.__w, u, ml, mr)

    def add_implicit_advection(self, kappa: Any = 1, marker = 0, f_id = 0):
        """Adds implicit advection physics to the variational form."""
        self.tracer_forms[f_id] += kappa*self.advection(self.__w, self.__u, marker)

    def add_implicit_downwind_advection(self, kappa: Any = 1, marker = 0, f_id = 0):
        self.tracer_forms[f_id] += kappa*self.downwind_advection(self.__w, self.__u, marker)

    def add_implicit_centered_advection(self, kappa: Any = 1, marker = 0, f_id = 0):
        self.tracer_forms[f_id] += kappa*self.centered_advection(self.__w, self.__u, marker)

    def add_explicit_diffusion(self, component_name, u, kappa: Any = 1, marker = 0, f_id = 0):
        """Adds explicit diffusion physics to the variational form."""

        D = self.molecular_diffusivity
        idx = self.component_dict[component_name]

        self.tracer_forms[f_id] += kappa*self.diffusion(self.__w[idx], u[idx], D[idx], marker)

    def add_implicit_diffusion(self, component_name, kappa: Any = 1, marker = 0, f_id = 0):
        """Adds implicit diffusion physics to the variational form."""

        D = self.molecular_diffusivity
        idx = self.component_dict[component_name]

        self.tracer_forms[f_id] += kappa*self.diffusion(self.__w[idx], self.__u[idx], D[idx], marker)

    def add_explicit_charge_balanced_diffusion(self, u, kappa: Any = 1, marker = 0, f_id = 0):
        self.tracer_forms[f_id] += kappa*self.charge_balanced_diffusion(self.__w, u, u, marker)

    def add_semi_implicit_charge_balanced_diffusion(self, u, kappa: Any = 1, marker = 0, f_id = 0):
        self.tracer_forms[f_id] += kappa*self.charge_balanced_diffusion(self.__w, self.__u, u, marker)

    def add_implicit_charge_balanced_diffusion(self, kappa: Any = 1, marker = 0, f_id = 0):
        self.tracer_forms[f_id] += kappa*self.charge_balanced_diffusion(self.__w, self.__u, self.__u, marker)

    def add_electric_field_advection(self, u, kappa: Any = 1, marker = 0, f_id = 0):
        D = self.molecular_diffusivity
        z = self.charge

        E = as_vector([z[i]*D[i]*self.electric_field for i in range(self.num_component)])

    def add_flux_limiter(self, u, u_up, k=-1.0, kappa: Any = 1, marker = 0, f_id = 0):
        """Sets up the components for flux limiters and add them to form."""

        self.tracer_forms[f_id] += \
        kappa*self.advection_flux_limited(self.__w, u, u_up, k, marker)

    def add_implicit_flux_limiter(self, u0, u_up, kappa: Any = 1, marker = 0, f_id = 0):
        # Consider renaming this function.

        self.tracer_forms[f_id] += \
        kappa*self.advection_implicit_flux_limited(self.__w, self.__u, u0, u_up, marker)

    def get_upwind_form(self, u):
        """"""

        # Consider renaming this function.

        adv = self.fluid_velocity
        n = self.n

        adv_np = (dot(adv, n) + Abs(dot(adv, n))) / 2.0
        adv_nm = (dot(adv, n) - Abs(dot(adv, n))) / 2.0

        np = sign(adv_np)
        nm = sign(adv_nm)

        upwind_form = dot(jump(nm*self.__w), jump(np*u/self.facet_area))*self.dS

        return upwind_form

    def add_dispersion(self):
        return #TODO: Setup this method.

    def add_mass_source(self, component_names, sources, kappa: Any = 1, f_id = 0):
        """Adds mass source to the variational form by component names."""

        for i, component_name in enumerate(component_names):
            idx = self.component_dict[component_name]
            self.tracer_forms[f_id] -= kappa*self.__w[idx]*sources[i]*self.dx

    def add_sources(self, sources, kappa: Any = 1, f_id = 0):
        self.tracer_forms[f_id] -= kappa*dot(self.__w, sources)*self.dx

    def get_forms(self):
        return self.tracer_forms

    def get_dirichlet_bcs(self):
        return self.__dirichlet_bcs

    def save_to_file(self, time, is_saving_pv=False):
        """"""

        try:
            self.outputter
        except:
            return False

        for i in range(self.num_component):
            self.write_function(self.fluid_comp_sub[i], time)

        if is_saving_pv:
            self.save_fluid_pressure(time)
            self.save_fluid_velocity(time)

        return True