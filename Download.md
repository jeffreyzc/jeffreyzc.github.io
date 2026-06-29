---
layout: page
title: Resources
---

Dear colleagues:

This page collects small Python tools I built for my own research workflow. Most of them were vibecoded with AI. They require editing the user-settings section before use, especially file paths, beamline geometry, detector center, q ranges, and experiment bounds. Please speak with beamline scientists and check your data folder for beamline-specific details needed to process your data.


<br>


### Thin film deposition tools
#### PLD Bayesian optimization helper and experiment logbook

**File:** [`pld_bayesian_optimizer_csv.py`](/assets/code/thinfilmbayesian.py)

A CSV-only Bayesian optimization helper for PLD experiment planning and logging. It reads completed experiments from a CSV log, rebuilds an Ax optimizer, suggests one new PLD recipe, and appends it as a pending trial.

This is an experimental tool using the AX platform. Give it a try - it can help if you have specific optimization goals.  

<br>

<br>

### Synchrotron data processing tools

#### 3D diffraction integration (for 3D synchrotron Surface Micro-diffraction and diffraction patterns, such as those obtained at APS 33-ID)

**File:** [`threeD_diffraction_integrate.py`](/assets/code/3D_diffraction_integrate.py)

Reads a 3D reciprocal-space `.vti` dataset, reconstructs the original \(Q_x, Q_y, Q_z\) grid, integrates selected windows, and exports 2D projections plus 1D line integrations.

<br>

#### 2D diffraction FWHM analysis (for general synchrotron X-ray diffraction images)

**File:** [`twoD_diffraction_fwhm.py`](/assets/code/2D_diffraction_FWHM.py)

Processes a series of 2D diffraction TIFF images, extracts 1D peak profiles, normalizes intensity, calculates peak position and FWHM, and tracks FWHM/strain evolution over time.

<br>

#### SAXS radial integration (for data obtained at the APS 12-ID-B beamline)

**File:** [`saxs_radial_integration.py`](/assets/code/saxs_radius_intergration.py)

Loads a 2D SAXS detector image from an HDF5 file, converts detector pixels into reciprocal-space coordinates, displays the image in q-space, overlays a selected q-annulus, and calculates a radial \(I(q)\) profile.

<br>
<br>

#### SAXS azimuthal integration (for data obtained at the APS 12-ID-B beamline)

**File:** [`saxs_azimuthal_integration.py`](/assets/code/saxs_azimuthal_integration.py)

Performs azimuthal integration within a selected q-annulus from a 2D SAXS image. It outputs \(I(phi)\), supports masking of bad pixels and beamstop/rod regions, and can save the integrated profile as an `.xy` file.

<br>










