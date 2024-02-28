# NDLArResponseSim

This is a package for modelling the current response of front-end electrodes within the DUNE ND-LAr detector.  This package aims to model the drift and weighting fields near the detector's anode plane, tabulate the current induced by charges drifting according to these calculations, and package these calculations in a form which can be integrated into larger models of the detector.

## Field Modelling

The primary tool used for field modelling is [Elmer FEM](https://www.elmerfem.org/blog/) 