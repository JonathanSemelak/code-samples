"""
container.py - Custom BuiltinModel for Orbital-Descriptor-Based Featurization
    
This module defines ExCorrModel, a subclass of TorchANI's BuiltinModel, which overrides the forward pass
to support orbital-based atomic environment vectors (orbital AEVs) computed via ExCorrAEVComputer.

The purpose of ExCorrModel is to serve as a custom model wrapper that integrates with a modified featurization pipeline using electron-density-based descriptors
The orbitalAEVComputer is assigned externally as aev_computer, typically during model assembly (via a custom assembler that sets ExCorrAEVComputer as the featurizer).

Notes: 
------
1. In the ANI context, AEV stands for Atomic Environment Vector. These are constructed from atom-centered symmetry
   functions and encode the geometric local environment in a way that is invariant to translation and rotation.
2. Orbital AEVs extend this idea by incorporating local electronic structure information, derived from orbital
   coefficients. The geometric component can be optionally included or omitted depending on the modeling goal
"""

import typing as tp

from torch import Tensor

from torchani.models import BuiltinModel
from torchani.tuples import SpeciesEnergies

class ExCorrModel(BuiltinModel):
    # Overloads forward to add "coefficients" input, and typing may not like it
    # since the signature differs from BuiltinModel.forward(), so we ignore typing here.

    # Init doesn't need anything else for now
    def forward(  # type: ignore
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        coefficients: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        """Calculates predicted energies for minibatch of configurations

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: tuple of tensors, species and energies for the given configurations
        """
        species_coordinates = self._maybe_convert_species(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, coefficients=coefficients, cell=cell, pbc=pbc)
        species, energies = self.neural_networks(species_aevs)
        energies += self.energy_shifter(species)
        return SpeciesEnergies(species, energies)
