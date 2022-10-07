# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *

def reaktoro_solve_rates(C_vector, C_scale, t_scale, reactions, state0):
    # input:
    # C_vector:           a numpy array of concentrations
    # t_scale:            The time scale that exchanges information between the dimensional and nondimensional worlds
    # C_scale:            The concentration scale
    # reactions:          The ReactionSystem class from reaktoro
    # state0:             The reaktoro state class

    # output:
    # reaction_rates:     the reaction rate solved by given concentration

    reaction_rates = np.zeros_like(C_vector)

    #C_scale = 6415.35 #mol/m3

    for i, c in enumerate(C_vector*C_scale):
        state0.setSpeciesAmount('NaCl(aq)', c, 'mol')
        state_properties = state0.properties()

        #print(state_properties.phaseDensities().val)

        CQ = rkt.ChemicalQuantity(state0)

        reaction_rates[i] = reactions.rates(state_properties).val #/CQ.value('phaseVolume(Aqueous)') # mols per second?

        # Ignoring the fact that density changes. Yoohoo!
        # C_vector_solved[i] = state0.speciesAmount('NaCl(aq)', 'mol') # /CQ.value("fluidVolume(units=m3)")

    return reaction_rates/C_scale*t_scale
