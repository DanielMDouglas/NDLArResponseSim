# NDLArResponseSim

This is a package for modelling the current response of front-end electrodes within the DUNE ND-LAr detector.  This package aims to model the drift and weighting fields near the detector's anode plane, tabulate the current induced by charges drifting according to these calculations, and package these calculations in a form which can be integrated into larger models of the detector.

## Field Modelling

The primary tool used for field modelling is [Elmer FEM](https://www.elmerfem.org/blog/), an open source multiphysical simulation software.  Other solver systems should produce identical results within this workflow, but these tools were primarily written with Elmer in mind.

First, a BREP model of a repeating unit of the pixel geometry in the region near the anode plane (from the back surface of the anode plane PCB to 12mm away from the from surface of that PCB).  The coordinate system used by these tools is such that z = 0 defines the LAr-facing sid of the anode back-plane (so that the back surface of the pixel pad is at z = 0).  The x and y coordinates are then parallel to the back-plane surface, such that the pixel edges are square to the x-y axes.

The shape of the pixels are defined by the following parameters:

| Pixel Pitch     | 3.8 mm | 4.434 mm |
| --------------- | ------ | -------- |
| Pixel width     | 3.0 mm | 3.5 mm   |
| Corner Radius   | 0.6 mm | 0.7 mm   |
| Pixel thickness | 40 um  | 40 um    |

The material properties used as inputs for the field solving steps are:

| Material Property         | Value |
| ------------------------- | ----- |
| LAr relative permittivity | 1.505 |

### Drift Field

The drift field is the usual electric field, defined by the boundary conditions and material properties of the detector volume.  The symmetries of the drift field (assuming it is regular even close to the edges of the drift volume) allow for solving the drift field in only one quarter of the pixel surface, with some extent in the drift direction (z).  In this volume, the electric field across the constant-x and constant-y surfaces is zero.  The backplane-facing surface boundary conditions are defined to be zero volts (ground) on the surface of the pixel, while the potential on the surface of the backplane is less well defined, but assumed to be some relatively large negative number (see discussion below).  The other constant-z surface assumes an electric field of 500 V/cm, the electric field in the bulk of the drift volume.

#### Backplane Potential

The true potential along the backplane surface is best found with a semi-dynamical simulation of current flux at the surface.  In this approach, both the pixel and backplane should be initialized at ground potential.  A static electric field should be found using the above methods.  From this field, one can evaluate the current into the surface of the relatively resistive backplane, which corresponds to an increment of charge for a single, small time-step.  The resulting charge distribution can be incorporated into the boundary conditions of the field solver, and a new electric field can be found.  With each time-step, charge paths will tend to diverge from the center of the exposed backplane surfaces, instead landing on the conductive pads. This operation can be repeated until the total current into the backplane is zero, at which point, the system is fully "charged up", and has reached a steady state.

Instead, this work assumes a constant potential along the backplane surface.  The value of this potential is found by evaluating the drift trajectories starting from a set of grid points and tracking their termination points.  By increasing the potential difference between the pixel pad and the backplane surface, these drift paths will either terminate on the pad or on the backplane itself.  The ultimate potential used in the production simulation is chosen as the minimum potential difference for which all drift paths terminate on the pixel pad.  This is a condition expected in the steady-state solution of the charge-up simulation described above, however the exact charge distribution on the backplane will be quite different, which may produce different drift path shapes.

### Weighting Field

The weighting field is a concept defined by the Ramo theorem, which frames the charge induced on an electrode in the vicinity of a moving charge similarly to an image charge problem.  WRITE EQUATIONS HERE

## Drift Simulation

A point-like charge bundle is simulated using a drift model SHOW DRIFT MODEL, CITE

A utility is included for parsing and interpolating field solver results from various solver packages.  

A utility is included for reading and interpolating Elmer results in the vtk format.  The drift path integration is done with a simple Euler integration method with a small timestep compared to the final tabulated time resolution.

## Response Packaging

### Far-field Extension

Since this package only covers the near-field simulation, the effects of induction from charges far from the anode need to be modelled seperately.  The simplest way to address this is to use a dipole approximation

### Tabulated Response

Shove it all into a numpy array with a given sample spatial distance (1/10 of pixel width) and time interval (0.1 us (?)) 

### Neural Surrogate Model

Use SIREN to produce a nueral surrogate model