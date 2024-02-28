# NDLArResponseSim

This is a package for modelling the current response of front-end electrodes within the DUNE ND-LAr detector.  This package aims to model the drift and weighting fields near the detector's anode plane, tabulate the current induced by charges drifting according to these calculations, and package these calculations in a form which can be integrated into larger models of the detector.

## Field Modelling

The primary tool used for field modelling is [Elmer FEM](https://www.elmerfem.org/blog/), an open source multiphysical simulation software.  First, a BREP model of a repeating unit of the pixel geometry in the region near the anode plane (from the back surface of the anode plane PCB to 12mm away from the from surface of that PCB).

### Drift Field

The drift field is the usual electric field, defined by the boundary conditions and material properties of the detector volume

### Weighting Field

The weighting field is a concept defined by the Ramo theorem, which frames the charge induced on an electrode in the vicinity of a moving charge similarly to an image charge problem.  WRITE EQUATIONS HERE

## Drift Simulation

A point-like charge bundle is simulated using a drift model SHOW DRIFT MODEL, CITE

## Response Packaging

### Tabulated Response

Shove it all into a numpy array with a given sample spatial distance (1/10 of pixel width) and time interval (0.1 us (?)) 

### Neural Surrogate Model

Use SIREN to produce a nueral surrogate model