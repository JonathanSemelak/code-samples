#!/usr/bin/env python3
"""
orbital_aev.py - Custom Feature Generator for Electron Density-Based Descriptors

This module defines the class `OrbitalAEVComputer`, which extracts and processes s, p, and d orbital coefficients
to generate descriptors for machine-learning interatomic potentials. It implements the feature engineering logic
required to feed ANI-style neural networks with information derived from the electronic density.

The main idea is to take orbital coefficients from the so-called "fitting density", associated with Cartesian-based
basis functions, and transform them into descriptor vectors that are:
1. Invariant to translations and rotations.
2. Expanded in a set of Gaussian functions, mimicking the atom-centered symmetry functions used in
   Behler-Parrinello-type models and adopted in ANI. This expansion is generally more effective for training
   neural networks than using raw or normalized signature vectors alone.

The feature engineering is still under development, but the current implementation supports:
- Simple orbital AEVs based on normalized norms of the coefficients.
- Full expansions with radial and angular contributions, analogous to geometric AEVs.
- Optional inclusion of geometric AEVs for hybrid representations.

Notes:
------
1. In the ANI context, AEV stands for Atomic Environment Vector. These are constructed from atom-centered symmetry
   functions and encode the geometric local environment in a way that is invariant to translation and rotation.
2. Orbital AEVs extend this idea by incorporating local electronic structure information, derived from orbital
   coefficients. The geometric component can be optionally included or omitted depending on the modeling goal.
"""

import typing as tp
import torch
from torch import Tensor
import numpy as np
from torchani.utils import ChemicalSymbolsToInts

# This calculates ONLY the coefficients part of the AEV
class OrbitalAEVComputer(torch.nn.Module):
    def forward(
        self,
        coefficients: Tensor,
        normalization_library: Tensor,
        species: Tensor,
        basis_functions: str,
        use_simple_orbital_aev: bool,
        use_angular_info: bool,
        use_angular_radial_coupling: bool,
        NOShfS = int,
        NOShfR = int,
        NOShfA = int,
        NOShfTheta = int,
        LowerOShfS = float,
        UpperOShfS = float,
        LowerOShfR = float,
        UpperOShfR = float,
        LowerOShfA = float,
        UpperOShfA = float,
        LowerOShfTheta = float,
        UpperOShfTheta = float,
        OEtaS = float,
        OEtaR = float,
        OEtaA = float,
        OZeta = float,
    ) -> Tensor:
        # We first need to reshape the coefficients for their manipulation
        # The s-type coefficients are processed with a custom radial AEV computer
        # The p and d-type coefficientes form an orbital matrix that basically
        # Describes the coordinates in the "space of coefficients" of fake atoms
        # Around each actual atom.
        
        s_coeffs, orbital_matrix = self._reshape_coefficients(coefficients,basis_functions)
        Z_to_idx = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}

        #Normalizes s_coeffs
        # Convert species -> row indices
        species_idx = self._map_species_to_idx(species, Z_to_idx)  # shape (nconf, natoms)

        # Unpack normalization library
        s_coeffs_mus, s_coeffs_sigmas, p_norms_mus, p_norms_sigmas = normalization_library   # unpack
        
        # Trims p normalization values to make them match other stuff
        p_norms_mus=p_norms_mus[:,:4]
        p_norms_sigmas = p_norms_sigmas[:,:4]
            
        s_coeffs = self._normalize(s_coeffs,species_idx,s_coeffs_mus,s_coeffs_sigmas)

        # Return "simple_orbital_aevs" if corresponding
        if use_simple_orbital_aev:
            # Case s
            if basis_functions == 's':
                return s_coeffs
            # Case sp or spd                
            p_norms = torch.linalg.norm(orbital_matrix, dim=-1)

            p_norms = self._normalize(p_norms, species_idx, p_norms_mus, p_norms_sigmas)

            simple_orbital_aevs = torch.cat((s_coeffs, p_norms), dim=-1)
            if use_angular_info:
                angles = self._get_angles_from_orbital_matrix(orbital_matrix,p_norms,True)
                simple_orbital_aevs = torch.cat((simple_orbital_aevs, angles), dim=-1)
            return simple_orbital_aevs  # shape (nconf, natoms, simple_orbital_aevs_length)
        
        else: # Return actual orbital_aevs
            nconf, natoms, nscoeffs = s_coeffs.shape
            # Define s shifts and reshape for broadcasting
            OShfS = torch.linspace(LowerOShfS, UpperOShfS, NOShfS)
            OShfS = OShfS.view(1, 1, 1, NOShfS)

            # s_coeffs are already normalized here, so we only prepare the tensor for broadcasting
            s_coeffs = s_coeffs.view(nconf, natoms, nscoeffs, 1)      

            # Compute the s component of the orbital AEV
            s_orbital_aev = torch.exp(-OEtaS * ((s_coeffs - OShfS) ** 2))
            
            # Reshapes
            s_orbital_aev = s_orbital_aev.view(nconf,natoms,nscoeffs*NOShfS)

            if basis_functions == 's':
                return s_orbital_aev

            p_norms = torch.linalg.norm(orbital_matrix, dim=-1)
            
            p_norms = self._normalize(p_norms, species_idx, p_norms_mus, p_norms_sigmas)

            _, _, np_norms = p_norms.shape                  
            # Define r shifts and reshape for broadcasting
            OShfR = torch.linspace(LowerOShfR, UpperOShfR, NOShfR)
            OShfR = OShfR.view(1, 1, 1, NOShfR)
            # Ensure the tensor is correctly shaped for broadcasting
            p_norms_exp = p_norms[..., None]          # add dim, no data copy
            # p_norms = p_norms.view(nconf, natoms, np_norms, 1)
            # Compute the squared differences
            radial_orbital_aev = torch.exp(-OEtaR * ((p_norms_exp - OShfR) ** 2))
            # Concatenate the s and radial contributions to the orbital aevs

            # Reshapes
            radial_orbital_aev = radial_orbital_aev.view(nconf,natoms,np_norms*NOShfR)

            orbital_aev = torch.cat((s_orbital_aev, radial_orbital_aev), dim=-1)            
            if use_angular_info:
                angles, avdistperangle = self._get_angles_from_orbital_matrix(orbital_matrix,p_norms,False)
                _, _, nangles = angles.shape
                print("nangles: ", nangles)
                # Define angle sections and reshape for broadcasting
                OShfTheta = torch.linspace(LowerOShfTheta, UpperOShfTheta, NOShfTheta)
                print("OShfTheta", OShfTheta)
                # Expand angles for ShfZ
                expanded_angles = angles.unsqueeze(-1)  # Adding an extra dimension for broadcasting ShfZ
                expanded_angles = expanded_angles.expand(-1, -1, -1, NOShfTheta)  # Explicitly expand to match ShfZ

                # Expand ShfZ to match angles
                OShfTheta = OShfTheta.view(1, 1, 1, NOShfTheta)  # Reshape for broadcasting
                OShfTheta = OShfTheta.expand(nconf, natoms, nangles, NOShfTheta)  # Match dimensions
 
                # Calculate factor1
                angular_orbital_aev = ((1 + torch.cos(expanded_angles - OShfTheta)) / 2)**OZeta
                print("angular_orbital_aev shape:", angular_orbital_aev.shape)
                angular_orbital_aev = 2**(1-OZeta)*angular_orbital_aev.unsqueeze(-1)
                print("angular_orbital_aev shape:", angular_orbital_aev.shape)

                if (use_angular_radial_coupling):
                        # Define r shifts for the angular component of the AEV and reshape for broadcasting
                    OShfA = torch.linspace(LowerOShfA, UpperOShfA, NOShfA)  
                    # Expand avdistperangle for ShfA
                    expanded_avdistperangle = avdistperangle.unsqueeze(-1)  # Adding an extra dimension for ShfA
                    expanded_avdistperangle = expanded_avdistperangle.expand(-1, -1, -1, NOShfA)  # Match dimensions of ShfA

                    # Expand ShfA to match avdistperangle
                    OShfA = OShfA.view(1, 1, 1, NOShfA)  # Reshape for broadcasting
                    OShfA = OShfA.expand(nconf, natoms, nangles, NOShfA)  # Match dimensions

                    # Calculate factor2
                    factor2 = torch.exp(-OEtaA * (expanded_avdistperangle - OShfA)**2)

                    # Combine factors
                    angular_orbital_aev = angular_orbital_aev * factor2.unsqueeze(-1) #unsqueeze(3)?

                    # Reshape to the final desired shape
                    print(angular_orbital_aev.shape)
                    print(nconf, natoms, nangles * NOShfA * NOShfTheta)
                    angular_orbital_aev = angular_orbital_aev.reshape(nconf, natoms, nangles * NOShfA * NOShfTheta)

                angular_orbital_aev = angular_orbital_aev.reshape(nconf, natoms, nangles *NOShfTheta)

                orbital_aev = torch.cat((orbital_aev, angular_orbital_aev), dim=-1) 

        return orbital_aev
        
    def _reshape_coefficients(
        self,
        coefficients: Tensor,
        basis_functions: str,
    ) -> tp.Tuple[Tensor, Tensor]:
        """ Output: A tuple containing 2 tensors: one with the s-type coefficients,
        and another with the p and d-type coefficients in the form of a matrix.

        The obtained orbital matrix will look like:

        [p0x  p0y  p0z]
        ...
        [p3x  p3y  p3z]
        [d0xx d0yy d0zz]
        [d0zy d0zx d0xy]
        ...
        [d3xx d3yy d3zz]
        [d3zy d3zx d3xy]

        Where each row of this matrix is an Atomic Orbital Vector (AOV). This resembles the
        diff_vec tensor (for a single atom) fron the geometric AEVs.
        """
        nconformers, natoms, _ = coefficients.shape

        s_coeffs = coefficients[:, :, :9]    # Shape: (nconformers, natoms, 9)
        p_coeffs = coefficients[:, :, 9:21]  # Shape: (nconformers, natoms, 12)
        d_coeffs = coefficients[:, :, 21:]   # Shape: (nconformers, natoms, 24)

        if basis_functions == 's':
            return s_coeffs, torch.tensor([])
        
        # Reshape p_coeffs to make it easier to handle individual components
        p_coeffs_reshaped = p_coeffs.view(nconformers, natoms, 4, 3)  # Shape: (nconformers, natoms, 4, 3)

        if basis_functions == 'sp':
            return s_coeffs, p_coeffs_reshaped #In this case the orbital_matrix only have AOVs from p-type coeffients
        
        # If we are in the 'spd' case, the orbital_matrix includes also AOVs from d-type coefficients

        # Reshape d_coeffs to make it easier to handle individual components
        # Correcting the reordering of d_coeffs
        # Transformation [Dxx, Dxy, Dyy, Dzx, Dzy, Dzz] -> [Dxx, Dyy, Dzz, Dzy, Dzx, Dxy]
        # which maps to indices [0, 2, 5, 4, 3, 1] respectively
        d_coeffs_reshaped = d_coeffs.view(nconformers, natoms, 4, 6)  # Shape: (nconformers, natoms, 4, 6)
        d_coeffs_reshaped_reordered = d_coeffs_reshaped[:, :, :, [0, 2, 5, 4, 3, 1]]

        # Splitting into two groups: diagonal [Dxx, Dyy, Dzz] and off-diagonal [Dzy, Dzx, Dxy]
        d_diagonal = d_coeffs_reshaped_reordered[:, :, :, :3]
        d_off_diagonal = d_coeffs_reshaped_reordered[:, :, :, 3:]

        # Concatenate modified p and d coefficients to form the desired "matrix"                
        orbital_matrix = torch.cat([p_coeffs_reshaped, d_diagonal, d_off_diagonal], dim=2)  # Shape (nconformers, natoms, 12, 3)

        return s_coeffs, orbital_matrix


    def _normalize(
        self,
        coeffs: Tensor,
        species_idx: Tensor,
        mus: Tensor,
        sigmas: Tensor
    ) -> Tensor:
        
        # Advanced indexing to fetch the right mu and sigma for each atom
        atom_mus = mus[species_idx, :]
        atom_sigmas = sigmas[species_idx, :]
       
        coeffs_normalized = (coeffs - atom_mus) / atom_sigmas
        return coeffs_normalized

    def _get_angles_from_orbital_matrix(
        self,
        orbital_matrix: Tensor,
        distances: Tensor,
        use_simple_orbital_aev: bool,
    ) -> Tensor:
        nconformers, natoms, naovs = distances.shape
        # Normalize the vectors using the provided distances

        # Create a mask for the zero vectors in the orbital matrix
        zero_mask = (orbital_matrix.abs() < 1e-12).all(dim=-1)

        # Perform the normalization, avoid division by zero by using where
        orbital_matrix_normalized = torch.where(
        zero_mask.unsqueeze(-1),
        torch.zeros_like(orbital_matrix),
        orbital_matrix / distances.unsqueeze(-1)        
        )
           
        # Calculate angles between each vector and the following vectors
        nangles = int((naovs-1)*naovs/2)
        angles = torch.zeros((nconformers, natoms,nangles))
        k = 0
        if use_simple_orbital_aev:
            for i in range(naovs):
                for j in range(i+1, naovs):
                    cos_angles = torch.einsum('ijk,ijk->ij', orbital_matrix_normalized[:, :, i, :], orbital_matrix_normalized[:, :, j, :])
                    cos_angles = torch.clamp(cos_angles, -0.9999, 0.9999)
                    angles[:, :, k] = torch.acos(cos_angles)
                    k = k + 1
            return angles
        else:
            avdistperangle = torch.zeros((nconformers, natoms, nangles))        
            for i in range(naovs):
                for j in range(i+1, naovs):
                    cos_angles = torch.einsum('ijk,ijk->ij', orbital_matrix_normalized[:, :, i, :], orbital_matrix_normalized[:, :, j, :])
                    cos_angles = torch.clamp(cos_angles, -0.9999, 0.9999)
                    angles[:, :, k] = torch.acos(cos_angles)
                    avdistperangle[:, :, k] = (distances[:, :, i]+distances[:, :, j])/2.0
                    k = k + 1
            return angles,avdistperangle

    def _map_species_to_idx(
            self,
            species: torch.Tensor, 
            Z_to_idx: dict,
    )-> Tensor:
        """
        species: shape (nconformers, natoms)
        Returns a new tensor of same shape with the row index in [0..N-1].
        """
        # Initialize an integer tensor
        species_idx = torch.empty_like(species, dtype=torch.long)

        # For each known Z in Z_to_idx, fill in the appropriate row index
        for z, row in Z_to_idx.items():
            mask = (species == z)
            species_idx[mask] = row
        return species_idx
