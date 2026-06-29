import os
import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# Edit only this section
# ============================================================
# ----------------------------
# File path
# ----------------------------
path = r"YOUR DATA FILE PATH"

# ----------------------------
# Detector / beamline geometry
# Please get these parameters from the data folder,
# metadata file, or beamline scientist.
# ----------------------------
D = 2.0122              # detector distance, m
p = 0.075e-3            # detector pixel size, m/pixel
lam = 0.9322e-10        # X-ray wavelength, m

x0_param = 1459.5       # beam center x position, pixel
y0_param = 206.28       # beam center y position, pixel


# ----------------------------
# Target q range for azimuthal integration
# Approximate real-space feature size: d = 2π / q
# Example: q = 1.0 Å^-1 corresponds to d ≈ 6.28 Å
# ----------------------------
qmin = 0.02             # Å^-1
qmax = 0.05             # Å^-1

# ----------------------------
# Azimuthal integration controls
# ----------------------------
phi_start = 0           # degrees
phi_end = 360           # degrees
phi_bin_width = 1       # degrees per bin

# Median is more robust against hot pixels / cosmic rays.
# Mean may be useful if the detector image is already very clean.
integration_statistic = "median"   # options: "median" or "mean"

# ----------------------------
# 2D pattern plotting controls
# ----------------------------

show_2d_pattern = True

# Show selected q range as two white annulus boundary circles?
show_selected_q_range = True

# Plot 2D SAXS pattern in log or linear intensity scale.
# Options: "log" or "linear"
plot_2d_scale = "log"

# q-space display range for 2D pattern
# These only affect the displayed 2D pattern window.
qx_plot_min = -0.10
qx_plot_max =  0.10

qy_plot_min = -0.07
qy_plot_max =  0.03


color_percentile_low = 1
color_percentile_high = 99.5

# Manual color limits
manual_vmin = 1
manual_vmax = 3

# Save the 2D pattern figure?
save_2d_pattern = False
output_2d_pattern_name = "selected_annulus_2D_pattern.png"

# Color scale mode for 2D pattern
# Options:
# "percentile" = automatically use percentiles
# "manual"     = use manually defined vmin/vmax
color_scale_mode = "manual"

# Percentile color scaling
color_percentile_low = 1
color_percentile_high = 99.0



# ----------------------------
# Masking controls
# ----------------------------

# Remove zero/negative pixels and extremely high intensity pixels
use_bad_pixel_mask = True

# Percentile threshold for removing very bright pixels / hot spots.
# 99.99 is conservative. Lower values remove more pixels.
saturation_percentile = 99.99

# Remove the vertical beamstop / detector rod region
use_rod_mask = True

# Half-width of the vertical rod mask, in pixels.
# Example: 8 means remove pixels with abs(dx) <= 8.
rod_half_width_px = 8

# ----------------------------
# I(phi) plotting controls
# ----------------------------
normalize_intensity = True

plot_figsize = (8, 5)
plot_ylim = (0, 1.1)

show_grid = True
save_Iphi_plot = False
output_Iphi_plot_name = "I_phi_plot.png"

# ----------------------------
# XY output controls
# ----------------------------

save_xy_file = True

# The .xy file will be saved in the same folder as the input .h5 file.
# If this is None, the filename will be generated automatically.
#
# Example automatic filename:
# SILHD_NL_SA_50_50.0C_00288_00002_q_0.02_0.05_Iphi.xy
output_xy_name = None



# ============================================================
# END OF USER CONTROL PART
# ============================================================






# ============================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================


# ============================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================


# ============================================================
# Helper functions
# ============================================================

def get_output_paths(input_path, qmin, qmax):
    """
    Generate output paths in the same folder as the input h5 file.
    """

    folder = os.path.dirname(input_path)
    base = os.path.splitext(os.path.basename(input_path))[0]

    if output_xy_name is None:
        xy_name = f"{base}_q_{qmin:g}_{qmax:g}_Iphi.xy"
    else:
        xy_name = output_xy_name

    xy_path = os.path.join(folder, xy_name)
    pattern_path = os.path.join(folder, output_2d_pattern_name)
    iphi_plot_path = os.path.join(folder, output_Iphi_plot_name)

    return xy_path, pattern_path, iphi_plot_path


def prepare_2d_image(
    img,
    scale="log",
    color_scale_mode="percentile",
    low_pct=1,
    high_pct=99.5,
    manual_vmin=None,
    manual_vmax=None
):
    """
    Prepare SAXS image for 2D plotting and calculate color limits.

    scale = "log":
        plot log10(intensity + 1)

    scale = "linear":
        plot raw intensity

    color_scale_mode = "percentile":
        use robust percentile-based color limits

    color_scale_mode = "manual":
        use manual_vmin and manual_vmax
    """

    scale = scale.lower()
    color_scale_mode = color_scale_mode.lower()

    if scale == "log":
        img_plot = np.log10(np.clip(img, a_min=0, a_max=None) + 1)
        colorbar_label = r"$\log_{10}(I + 1)$"

    elif scale == "linear":
        img_plot = img.copy()
        colorbar_label = "Intensity"

    else:
        raise ValueError('plot_2d_scale must be either "log" or "linear".')

    finite_vals = img_plot[np.isfinite(img_plot)]

    if finite_vals.size == 0:
        raise ValueError("No finite values found in image for color scaling.")

    if color_scale_mode == "percentile":
        vmin = np.percentile(finite_vals, low_pct)
        vmax = np.percentile(finite_vals, high_pct)

    elif color_scale_mode == "manual":
        if manual_vmin is None or manual_vmax is None:
            raise ValueError(
                "manual_vmin and manual_vmax must be set when color_scale_mode = 'manual'."
            )
        vmin = manual_vmin
        vmax = manual_vmax

    else:
        raise ValueError('color_scale_mode must be either "percentile" or "manual".')

    return img_plot, vmin, vmax, colorbar_label


def calculate_I_phi(phi_vals, I_vals, bins_phi, integration_statistic):
    """
    Calculate azimuthal intensity profile I(phi).
    """

    nbins_phi = len(bins_phi) - 1
    I_phi = np.full(nbins_phi, np.nan)

    for i in range(nbins_phi):
        m = (phi_vals >= bins_phi[i]) & (phi_vals < bins_phi[i + 1])

        if np.any(m):
            if integration_statistic.lower() == "median":
                I_phi[i] = np.median(I_vals[m])
            elif integration_statistic.lower() == "mean":
                I_phi[i] = np.mean(I_vals[m])
            else:
                raise ValueError(
                    "integration_statistic must be either 'median' or 'mean'."
                )

    return I_phi


# ============================================================
# 1. Load SAXS image
# ============================================================

with h5py.File(path, "r") as f:
    img = f["entry/data/data"][...].astype(np.float64)

H, W = img.shape

# Convert beam center based on image coordinate convention
center = (x0_param, H - y0_param)

print("Loaded image:")
print(f"  File: {path}")
print(f"  Image shape: {H} rows × {W} columns")
print(f"  Beam center used in script: x = {center[0]:.2f}, y = {center[1]:.2f}")
print()


# ============================================================
# 2. Coordinate geometry
# ============================================================

y, x = np.indices(img.shape)

dx = x - center[0]
dy = y - center[1]

# Radial distance from beam center, in meters
r_m = np.hypot(dx, dy) * p

# Scattering angle
two_theta = np.arctan2(r_m, D)

# q in Å^-1
q_A = (4 * np.pi / lam) * np.sin(two_theta / 2) * 1e-10

# qx and qy in Å^-1
# This small-angle mapping is mainly used for display.
qx_A = dx * p / D * (2 * np.pi / lam) * 1e-10
qy_A = dy * p / D * (2 * np.pi / lam) * 1e-10

# Azimuthal angle phi, in degrees, from 0 to 360
phi = (np.degrees(np.arctan2(dy, dx)) + 360) % 360


# ============================================================
# 3. Masking and annulus selection
# ============================================================

# Select target q annulus
annulus = (q_A >= qmin) & (q_A <= qmax)

# Start with all pixels in annulus as good
good = annulus.copy()

# Bad pixel / hot pixel mask
if use_bad_pixel_mask:
    hi_thresh = np.percentile(img, saturation_percentile)
    bad_mask = (img <= 0) | (img >= hi_thresh)
    good &= ~bad_mask
else:
    hi_thresh = None
    bad_mask = np.zeros_like(img, dtype=bool)

# Vertical rod / beamstop mask
if use_rod_mask:
    rod_mask = np.abs(dx) <= rod_half_width_px
    good &= ~rod_mask
else:
    rod_mask = np.zeros_like(img, dtype=bool)


# Output paths
xy_path, pattern_path, iphi_plot_path = get_output_paths(path, qmin, qmax)


# Print masking summary
annulus_pixels = np.count_nonzero(annulus)
good_pixels = np.count_nonzero(good)

print("Masking summary:")
print(f"  q range: {qmin} to {qmax} Å^-1")
print(f"  Pixels in q annulus: {annulus_pixels:,}")
print(f"  Good pixels after masking: {good_pixels:,}")

if annulus_pixels > 0:
    print(f"  Fraction of annulus retained: {100 * good_pixels / annulus_pixels:.2f}%")
else:
    print("  Warning: no pixels found in selected q annulus.")

if use_bad_pixel_mask:
    print(f"  Saturation percentile: {saturation_percentile}")
    print(f"  High-intensity threshold: {hi_thresh:.3g}")

if use_rod_mask:
    print(f"  Rod mask half-width: {rod_half_width_px} pixels")

print()


# ============================================================
# 4. Show 2D SAXS pattern
# ============================================================

if show_2d_pattern:
    img_plot, vmin, vmax, colorbar_label = prepare_2d_image(
        img,
        scale=plot_2d_scale,
        color_scale_mode=color_scale_mode,
        low_pct=color_percentile_low,
        high_pct=color_percentile_high,
        manual_vmin=manual_vmin,
        manual_vmax=manual_vmax
    )
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot image in q-space.
    # qy is displayed in detector-style orientation:
    # negative qy appears near the top if qy_plot_min is negative.
    extent = [
        np.nanmin(qx_A),
        np.nanmax(qx_A),
        np.nanmax(qy_A),
        np.nanmin(qy_A)
    ]

    im = ax.imshow(
        img_plot,
        extent=extent,
        origin="upper",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="equal"
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

    # Draw qmin and qmax annulus boundaries only if requested
    if show_selected_q_range:
        theta = np.linspace(0, 2 * np.pi, 1000)

        ax.plot(
            qmin * np.cos(theta),
            qmin * np.sin(theta),
            "w-",
            lw=1.8
        )

        ax.plot(
            qmax * np.cos(theta),
            qmax * np.sin(theta),
            "w-",
            lw=1.8
        )

    # Mark beam center
    ax.plot(
        0,
        0,
        marker="+",
        color="white",
        markersize=12,
        markeredgewidth=2
    )

    # Separate qx and qy display limits
    ax.set_xlim(qx_plot_min, qx_plot_max)

    # Match detector-style display: negative qy appears toward the top.
    ax.set_ylim(qy_plot_max, qy_plot_min)

    ax.set_xlabel(r"$q_x$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$q_y$ ($\AA^{-1}$)")

    if show_selected_q_range:
        title_q_part = rf"Selected annulus: q = {qmin:g}–{qmax:g} $\AA^{{-1}}$"
    else:
        title_q_part = "2D SAXS pattern"

    ax.set_title(
        title_q_part
        + rf" ({plot_2d_scale.lower()} scale)"
        + "\ncenter fixed"
    )

    plt.tight_layout()

    if save_2d_pattern:
        plt.savefig(pattern_path, dpi=300)
        print("2D pattern figure saved to:")
        print(f"  {pattern_path}")
        print()

    plt.show()


# ============================================================
# 5. Azimuthal integration: I(phi)
# ============================================================

# Define phi bins
nbins_phi = int((phi_end - phi_start) / phi_bin_width)

if nbins_phi <= 0:
    raise ValueError("Invalid phi range or phi_bin_width. Check phi_start, phi_end, and phi_bin_width.")

bins_phi = np.linspace(phi_start, phi_end, nbins_phi + 1)
phi_centers = 0.5 * (bins_phi[:-1] + bins_phi[1:])

# Extract good pixels
phi_vals = phi[good].ravel()
I_vals = img[good].ravel()

if phi_vals.size == 0:
    raise ValueError(
        "No pixels left after q selection and masking. "
        "Try changing qmin/qmax or relaxing the masks."
    )

I_phi = calculate_I_phi(
    phi_vals=phi_vals,
    I_vals=I_vals,
    bins_phi=bins_phi,
    integration_statistic=integration_statistic
)


# Normalize intensity if requested
if normalize_intensity:
    max_I = np.nanmax(I_phi)

    if np.isfinite(max_I) and max_I > 0:
        I_phi = I_phi / max_I
    else:
        print("Warning: intensity could not be normalized because max intensity is invalid.")


# ============================================================
# 6. Save I(phi) as .xy file
# ============================================================

if save_xy_file:
    xy_data = np.column_stack([phi_centers, I_phi])

    header = (
        "Azimuthal integration I(phi)\n"
        f"Input file: {path}\n"
        f"q range: {qmin} to {qmax} A^-1\n"
        f"Detector distance D: {D} m\n"
        f"Pixel size p: {p} m/pixel\n"
        f"Wavelength lambda: {lam} m\n"
        f"Beam center x0_param: {x0_param} pixel\n"
        f"Beam center y0_param: {y0_param} pixel\n"
        f"Integration statistic: {integration_statistic}\n"
        f"Normalized intensity: {normalize_intensity}\n"
        "Columns:\n"
        "phi_degree intensity"
    )

    np.savetxt(
        xy_path,
        xy_data,
        fmt="%.8f %.8e",
        header=header,
        comments="# "
    )

    print("I(phi) .xy file saved to:")
    print(f"  {xy_path}")
    print()


# ============================================================
# 7. Plot I(phi)
# ============================================================

plt.figure(figsize=plot_figsize)

plt.plot(
    phi_centers,
    I_phi,
    "k-",
    lw=2,
    label=f"{integration_statistic.capitalize()} intensity"
)

plt.xlim(phi_start, phi_end)

if plot_ylim is not None:
    plt.ylim(plot_ylim)

plt.xlabel(r"Azimuth $\phi$ ($^\circ$)")

if normalize_intensity:
    plt.ylabel("Normalized Intensity (a.u.)")
else:
    plt.ylabel("Intensity (a.u.)")

plt.title(
    rf"$I(\phi)$ for $q = {qmin:g}-{qmax:g}$ $\AA^{{-1}}$"
    "\nRestricted to Good Detector Region"
)

if show_grid:
    plt.grid(True, alpha=0.2)

plt.legend()
plt.tight_layout()

if save_Iphi_plot:
    plt.savefig(iphi_plot_path, dpi=300)
    print("I(phi) plot saved to:")
    print(f"  {iphi_plot_path}")
    print()

plt.show()


# ============================================================
# 8. Print approximate real-space feature range
# ============================================================

# d = 2π / q
# Because q is in Å^-1, d is in Å.
d_max = 2 * np.pi / qmin
d_min = 2 * np.pi / qmax

print("Approximate real-space feature size range:")
print(f"  qmin = {qmin} Å^-1  ->  d ≈ {d_max:.2f} Å")
print(f"  qmax = {qmax} Å^-1  ->  d ≈ {d_min:.2f} Å")
print(f"  Feature range: approximately {d_min:.2f}–{d_max:.2f} Å")
