"""
SAXS 2D Image Visualization and Radial q-Integration
====================================================

This script loads a 2D SAXS detector image from an HDF5 file, converts the
detector pixel coordinates into reciprocal-space coordinates, displays the
image in q-space, overlays a selected q-annulus, and performs a radial I(q)
integration within that selected annulus.

Required packages
-----------------
pip install h5py hdf5plugin numpy matplotlib

Notes
-----
1. The detector center values should come from your beamline/setup parameter file.
2. Some beamline parameter files use a different y-coordinate convention than
   the image array. This script assumes the center from the parameter file needs
   the y-coordinate flipped using:

       y0_image = image_height - y0_param

   If your beam center appears wrong, check this convention first.
3. The q-annulus should be entered using positive q values in Å⁻¹.
"""

# ============================================================
# USER SETTINGS
# ============================================================

# ----------------------------
# File path
# ----------------------------
# Replace this with your own HDF5 data file path.
DATA_PATH = r"YOUR DATA FILE LOCATION"

# ----------------------------
# HDF5 dataset location
# ----------------------------
# Common Eiger/Dectris-style path:
DATASET_PATH = "entry/data/data"

# ----------------------------
# SAXS geometry
# ----------------------------
D = 2.0122          # sample-detector distance, in meters
p = 0.075e-3        # detector pixel size, in meters/pixel
lam = 0.9322e-10    # X-ray wavelength, in meters

# Beam center from setup/parameter file
# x0 is usually directly usable.
# y0 may need to be flipped depending on the beamline convention.
x0_param = 1459.5
y0_param = 206.28

# ----------------------------
# q-annulus selection
# ----------------------------
# Select the radial q range to integrate, in Å^-1.
# Use positive values.
qmin = 0.02
qmax = 0.05

# ----------------------------
# Mask settings
# ----------------------------
beamstop_r = 40       # beamstop radius, in pixels
rod_hw = 6            # half-width of vertical beamstop rod, in pixels
hi_pct = 99.9995      # percentile threshold for saturated pixels

# ----------------------------
# Plot display settings
# ----------------------------
vmin = 0.5
vmax = 3.0

xrange_q = (-0.10, 0.10)
yrange_q = (-0.075, 0.025)

cmap = "viridis"

# ----------------------------
# Integration settings
# ----------------------------
nbins_q = 400
clip_sigma = 4        # robust sigma clipping for I(q); set to None to disable


# ============================================================
# NON-USER SETTINGS / FUNCTIONS
# Usually you do not need to edit below this line.
# ============================================================

import hdf5plugin
import h5py
import numpy as np
import matplotlib.pyplot as plt


def load_hdf5_image(path, dataset_path):
    """
    Load a 2D image from an HDF5 file.

    If the dataset contains a stack of images, this function uses the first frame.
    Modify this behavior if you need to average or select a specific frame.
    """
    with h5py.File(path, "r") as f:
        data = f[dataset_path][...]

    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 2:
        img = data
    elif data.ndim == 3:
        img = data[0]
        print(f"Dataset is 3D with shape {data.shape}. Using the first frame.")
    else:
        raise ValueError(f"Expected a 2D image or 3D image stack, but got shape {data.shape}.")

    return img


def bad_mask_components(img, center, beamstop_r=40, rod_hw=6, hi_pct=99.9995):
    """
    Create individual bad-pixel mask components.

    Components:
    - beamstop disk
    - vertical beamstop rod
    - zero/negative pixels
    - saturated/high-intensity pixels
    """
    y, x = np.indices(img.shape)

    dx = x - center[0]
    dy = y - center[1]
    r = np.hypot(dx, dy)

    m_beam = r < beamstop_r
    m_rod = np.abs(dx) <= rod_hw

    hi = np.percentile(img, hi_pct)
    m_bad = (img <= 0) | (img >= hi)

    return m_beam, m_rod, m_bad


def binned_stat(x, y, bins, stat="median", clip_sigma=None):
    """
    Robust 1D binning.

    Parameters
    ----------
    x : array
        Coordinate values, for example q.
    y : array
        Intensity values.
    bins : array
        Bin edges.
    stat : str
        "median" or "mean".
    clip_sigma : float or None
        If provided, applies robust MAD-based sigma clipping in each bin.

    Returns
    -------
    out : array
        Binned intensity values.
    """
    out = np.full(len(bins) - 1, np.nan, dtype=float)

    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i + 1])

        if not np.any(m):
            continue

        vals = y[m]

        if clip_sigma is not None:
            med = np.median(vals)
            mad = np.median(np.abs(vals - med)) + 1e-12
            z = 0.6745 * (vals - med) / mad
            vals = vals[np.abs(z) <= clip_sigma]

            if vals.size == 0:
                continue

        if stat == "median":
            out[i] = np.median(vals)
        elif stat == "mean":
            out[i] = np.mean(vals)
        else:
            raise ValueError("stat must be either 'median' or 'mean'.")

    return out


# ============================================================
# LOAD DATA
# ============================================================

img = load_hdf5_image(DATA_PATH, DATASET_PATH)
H, W = img.shape

print(f"Loaded image shape: {img.shape}")


# ============================================================
# BEAM CENTER
# ============================================================

# Convert beam center from parameter-file convention to image-array convention.
center = (x0_param, H - y0_param)

print(f"Beam center used in image coordinates: x = {center[0]:.2f}, y = {center[1]:.2f}")


# ============================================================
# COORDINATE MAPS
# ============================================================

y, x = np.indices(img.shape)

dx = x - center[0]
dy = y - center[1]

# Small-angle linearized qx and qy, in Å^-1
qx = (2 * np.pi / lam) * (dx * p / D) * 1e-10
qy = (2 * np.pi / lam) * (dy * p / D) * 1e-10

# Exact radial q using detector radius and scattering angle
r_pix = np.hypot(dx, dy)
r_m = r_pix * p

two_theta = np.arctan2(r_m, D)
q_A = (4 * np.pi / lam) * np.sin(two_theta / 2) * 1e-10


# ============================================================
# MASK AND ANNULUS
# ============================================================

qmin, qmax = sorted((abs(qmin), abs(qmax)))

annulus = (q_A >= qmin) & (q_A <= qmax)
n_annulus = int(np.count_nonzero(annulus))

print(f"Selected q range: {qmin:.4f} to {qmax:.4f} Å^-1")
print(f"Pixels in annulus before masking: {n_annulus}")

m_beam, m_rod, m_bad = bad_mask_components(
    img,
    center,
    beamstop_r=beamstop_r,
    rod_hw=rod_hw,
    hi_pct=hi_pct,
)

good = annulus & (~(m_beam | m_rod | m_bad))
n_good = int(np.count_nonzero(good))

print(f"Pixels in annulus after full mask: {n_good}")


# ------------------------------------------------------------
# Optional automatic mask relaxation for very low-q annuli
# ------------------------------------------------------------

if n_good < 200:
    q_beam = (2 * np.pi / lam) * ((beamstop_r * p) / D) * 1e-10
    print(f"Estimated q blocked by beamstop: ~{q_beam:.4f} Å^-1")

    relax_beam = qmax <= 1.20 * q_beam

    if relax_beam:
        print("Low-q annulus is near the beamstop. Relaxing beamstop mask.")
        good = annulus & (~(m_bad | m_rod))
        n_good = int(np.count_nonzero(good))
        print(f"Pixels after relaxing beamstop mask: {n_good}")

    if n_good < 200:
        print("Still few pixels. Relaxing rod mask as well, keeping only bad-pixel mask.")
        good = annulus & (~m_bad)
        n_good = int(np.count_nonzero(good))
        print(f"Pixels after relaxing rod mask: {n_good}")

if n_good == 0:
    raise RuntimeError(
        "No pixels left in the selected q range after masking. "
        "Try widening qmin/qmax or relaxing the mask settings."
    )


# ============================================================
# STEP 1 — SHOW SAXS IMAGE IN q-SPACE
# ============================================================

log_img = np.log10(img + 1)

plt.figure(figsize=(7, 6))

plt.imshow(
    log_img,
    extent=[qx.min(), qx.max(), qy.min(), qy.max()],
    origin="lower",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
)

plt.xlim(xrange_q)
plt.ylim(yrange_q)

# This keeps the displayed orientation consistent with the original script.
plt.gca().invert_yaxis()

plt.plot(0, 0, "w+", ms=12, mew=2)

plt.colorbar(label=r"log$_{10}$(Intensity + 1)")
plt.xlabel(r"$q_x$ ($\mathrm{\AA}^{-1}$)")
plt.ylabel(r"$q_y$ ($\mathrm{\AA}^{-1}$)")
plt.title("SAXS image in q-space")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 2 — OVERLAY SELECTED q-ANNULUS
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(
    log_img,
    extent=[qx.min(), qx.max(), qy.min(), qy.max()],
    origin="lower",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
)

plt.xlim(xrange_q)
plt.ylim(yrange_q)
plt.gca().invert_yaxis()

ax = plt.gca()

for qring in (qmin, qmax):
    ax.add_patch(
        plt.Circle(
            (0, 0),
            qring,
            color="w",
            fill=False,
            lw=1.4,
        )
    )

plt.plot(0, 0, "w+", ms=12, mew=2)

plt.colorbar(label=r"log$_{10}$(Intensity + 1)")
plt.xlabel(r"$q_x$ ($\mathrm{\AA}^{-1}$)")
plt.ylabel(r"$q_y$ ($\mathrm{\AA}^{-1}$)")
plt.title(f"Selected annulus: q = {qmin:.4f}–{qmax:.4f} Å⁻¹")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 3 — RADIAL I(q) INTEGRATION WITHIN SELECTED ANNULUS
# ============================================================

q_vals = q_A[good].ravel()
I_vals = img[good].ravel()

bins_q = np.linspace(qmin, qmax, nbins_q + 1)
q_centers = 0.5 * (bins_q[:-1] + bins_q[1:])

I_q = binned_stat(
    q_vals,
    I_vals,
    bins_q,
    stat="median",
    clip_sigma=clip_sigma,
)

plt.figure(figsize=(7, 5))

plt.plot(q_centers, I_q, "k-", lw=1.2)

plt.xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
plt.ylabel("Intensity, a.u.")
plt.title(f"I(q) within q = {qmin:.4f}–{qmax:.4f} Å⁻¹")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
