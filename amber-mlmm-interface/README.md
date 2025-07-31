# AMBER ML/MM Interface

This folder contains a customized implementation of **hybrid ML/MM (machine learning / molecular mechanics)** functionality within the AMBER simulation engine. It enables the use of neural network potentials (e.g., ANI or custom PyTorch models) as a drop-in replacement for the QM region in traditional QM/MM workflows.

## Motivation

ML/MM approaches aim to combine the speed of classical MD with the accuracy of machine-learned quantum models. This implementation builds upon AMBER’s standard QM/MM interface, replacing quantum mechanical calculations with calls to an external machine learning potential.

Key benefits include:
- Orders of magnitude faster than conventional QM/MM.
- Sophisticated corrections to ML-predicted gas-phase energies and forces via an electrostatic polarization model.

## Structure

The implementation is contained in a single Fortran source file, consistent with AMBER’s internal code structure. It defines multiple subroutines for:
- Model initialization and energy/force evaluation via a PyTorch backend.
- Applying polarization corrections to gas-phase ML predictions.
- Computing ML/MM Coulombic interactions (energies and forces) analytically.

## Notes
- This interface was officially merged into the Amber master branch and is included in the AmberTools25 release.
- This code was used in [this publication](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c01792) (which was selected for the [journal cover](https://pubs.acs.org/toc/jctcce/21/10)!)
- The underlying physical model was validated in [this study](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.4c00478).
- It depends on the [TorchaniAMBER interface](https://github.com/roitberg-group/torchani-amber), a C++ wrapper for PyTorch-based ANI models. My contributions to that repository were limited and focused mainly on documentation and validation. The AMBER-side implementation provided here is entirely my own work.
- Future plans include benchmarking on binding free energy calculations and fine-tuning the approach for specific use cases, such as conformational exploration and binding of macrocyclic peptides.

