#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import vtk
from vtk.util import numpy_support
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================================
# User settings
# ================================

# input VTI filename
vti_input_file = "xxxxx.vti"


verbose = True
plot_logscale = 0      # 0 = linear intensity, 1 = log10(1 + intensity)
write_file = 1
show_plots = True

dpi = 150
cmap = "gist_stern"
vmin = 0.35
vmax = 1.0

# Integration windows in the ORIGINAL VTI coordinate system, no transform
qx_pos, qx_delta = 1.2, 0.4
qy_pos, qy_delta = -1.2, 0.7
qz_pos, qz_delta = 1.698, 0.16

# Optional view limits for saved plots. Set to None to show full data range.
yz_xlim = (-1.9, -0.5)
yz_ylim = (1.6525, 1.75)
xz_xlim = (1.0, 1.45)
xz_ylim = (1.65, 1.75)
xy_xlim = (0.9, 1.5)
xy_ylim = (-1.9, -0.7)


# ================================
# Helper functions
# ================================

def find_indices(axis, low, high):
    """Return index range [i1:i2] closest to the requested coordinate window."""
    axis = np.asarray(axis)
    idx1 = int(np.abs(axis - low).argmin())
    idx2 = int(np.abs(axis - high).argmin())
    return min(idx1, idx2), max(idx1, idx2) + 1


def normalize(data):
    """Normalize data to 0-1 for plotting only."""
    data = np.asarray(data, dtype=float)
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)


def prep_for_display(data):
    """Apply optional log scale, then normalize for imshow/line plots."""
    data = np.asarray(data, dtype=float)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if plot_logscale:
        data = np.log10(1.0 + np.clip(data, a_min=0, a_max=None))
    return normalize(data)


def add_window_lines(ax, x_low=None, x_high=None, y_low=None, y_high=None):
    if x_low is not None and x_high is not None:
        ax.axvline(x_low, color="r", linestyle="--", linewidth=1)
        ax.axvline(x_high, color="r", linestyle="--", linewidth=1)
    if y_low is not None and y_high is not None:
        ax.axhline(y_low, color="r", linestyle="--", linewidth=1)
        ax.axhline(y_high, color="r", linestyle="--", linewidth=1)


# ================================
# Read VTI file
# ================================

vti_file = os.path.abspath(vti_input_file)
data_dir, filename = os.path.split(vti_file)
filebase, ext = os.path.splitext(filename)

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(vti_file)
reader.Update()
vti_data = reader.GetOutput()

vti_point_data = vti_data.GetPointData()
vti_array_data = vti_point_data.GetScalars()
if vti_array_data is None:
    raise RuntimeError("No scalar array found in the VTI file.")

array_data = numpy_support.vtk_to_numpy(vti_array_data)

dim = vti_data.GetDimensions()      # (nx, ny, nz)
steps = vti_data.GetSpacing()       # (dx, dy, dz)
origin = vti_data.GetOrigin()       # (x0, y0, z0)

# IMPORTANT: VTI grid coordinates use dim - 1 intervals, not dim intervals.
qx = origin[0] + np.arange(dim[0]) * steps[0]
qy = origin[1] + np.arange(dim[1]) * steps[1]
qz = origin[2] + np.arange(dim[2]) * steps[2]

# Convert VTK flat scalar array to data[x, y, z].
# VTK/Numpy gives z-y-x order after reshape; transpose to x-y-z.
data3d = np.reshape(array_data, dim[::-1]).transpose(2, 1, 0)

if verbose:
    print("VTI file:", vti_file)
    print("Dimensions (nx, ny, nz):", dim)
    print("Spacing:", steps)
    print("Origin:", origin)
    print(f"Qx range: {qx.min():.6f} to {qx.max():.6f}, n={len(qx)}")
    print(f"Qy range: {qy.min():.6f} to {qy.max():.6f}, n={len(qy)}")
    print(f"Qz range: {qz.min():.6f} to {qz.max():.6f}, n={len(qz)}")

# ================================
# Slice/integrate using original coordinates
# ================================

qx_low, qx_high = qx_pos - qx_delta / 2, qx_pos + qx_delta / 2
qy_low, qy_high = qy_pos - qy_delta / 2, qy_pos + qy_delta / 2
qz_low, qz_high = qz_pos - qz_delta / 2, qz_pos + qz_delta / 2

qx_ind1, qx_ind2 = find_indices(qx, qx_low, qx_high)
qy_ind1, qy_ind2 = find_indices(qy, qy_low, qy_high)
qz_ind1, qz_ind2 = find_indices(qz, qz_low, qz_high)

if verbose:
    print("Integration windows:")
    print(f"  Qx: requested {qx_low:.6f} to {qx_high:.6f}; indices {qx_ind1}:{qx_ind2}; actual {qx[qx_ind1]:.6f} to {qx[qx_ind2-1]:.6f}")
    print(f"  Qy: requested {qy_low:.6f} to {qy_high:.6f}; indices {qy_ind1}:{qy_ind2}; actual {qy[qy_ind1]:.6f} to {qy[qy_ind2-1]:.6f}")
    print(f"  Qz: requested {qz_low:.6f} to {qz_high:.6f}; indices {qz_ind1}:{qz_ind2}; actual {qz[qz_ind1]:.6f} to {qz[qz_ind2-1]:.6f}")

# 2D projections / slices:
# YZ = integrate through selected Qx range
# XZ = integrate through selected Qy range
# XY = integrate through selected Qz range
data_slice_yz = data3d[qx_ind1:qx_ind2, :, :].sum(axis=0)   # shape: (ny, nz)
data_slice_xz = data3d[:, qy_ind1:qy_ind2, :].sum(axis=1)   # shape: (nx, nz)
data_slice_xy = data3d[:, :, qz_ind1:qz_ind2].sum(axis=2)   # shape: (nx, ny)

# 1D integrations over the corresponding two-axis windows
line_x = data3d[:, qy_ind1:qy_ind2, qz_ind1:qz_ind2].sum(axis=(1, 2))
line_y = data3d[qx_ind1:qx_ind2, :, qz_ind1:qz_ind2].sum(axis=(0, 2))
line_z = data3d[qx_ind1:qx_ind2, qy_ind1:qy_ind2, :].sum(axis=(0, 1))

line_x_normalized = prep_for_display(line_x)
line_y_normalized = prep_for_display(line_y)
line_z_normalized = prep_for_display(line_z)

# ================================
# Output folder
# ================================

output_dir = os.path.join(data_dir, filebase)
os.makedirs(output_dir, exist_ok=True)

# ================================
# Plot 2D integrated slices
# ================================

# Figure 1: YZ projection
fig, ax = plt.subplots(figsize=(15, 8))
im = ax.imshow(
    prep_for_display(data_slice_yz).T,
    origin="lower",
    cmap=cmap,
    extent=(qy.min(), qy.max(), qz.min(), qz.max()),
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
)
ax.set_title(f"{filebase}: YZ projection, integrated over Qx")
ax.set_xlabel("Qy")
ax.set_ylabel("Qz")
add_window_lines(ax, x_low=qy_low, x_high=qy_high, y_low=qz_low, y_high=qz_high)
ax.add_patch(patches.Rectangle((qy_low, qz_low), qy_delta, qz_delta, linewidth=1, edgecolor="r", facecolor="none"))
fig.colorbar(im, ax=ax, label="Normalized intensity")
if yz_xlim is not None:
    ax.set_xlim(*yz_xlim)
if yz_ylim is not None:
    ax.set_ylim(*yz_ylim)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, f"{filebase}_yz_integrated_over_qx.tiff"), dpi=dpi, format="tiff")

# Figure 2: XZ projection
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(
    prep_for_display(data_slice_xz).T,
    origin="lower",
    cmap=cmap,
    extent=(qx.min(), qx.max(), qz.min(), qz.max()),
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
)
ax.set_title(f"{filebase}: XZ projection, integrated over Qy")
ax.set_xlabel("Qx")
ax.set_ylabel("Qz")
add_window_lines(ax, x_low=qx_low, x_high=qx_high, y_low=qz_low, y_high=qz_high)
ax.add_patch(patches.Rectangle((qx_low, qz_low), qx_delta, qz_delta, linewidth=1, edgecolor="r", facecolor="none"))
fig.colorbar(im, ax=ax, label="Normalized intensity")
if xz_xlim is not None:
    ax.set_xlim(*xz_xlim)
if xz_ylim is not None:
    ax.set_ylim(*xz_ylim)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, f"{filebase}_xz_integrated_over_qy.tiff"), dpi=dpi, format="tiff")

# Figure 3: XY projection
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(
    prep_for_display(data_slice_xy).T,
    origin="lower",
    cmap=cmap,
    extent=(qx.min(), qx.max(), qy.min(), qy.max()),
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
)
ax.set_title(f"{filebase}: XY projection, integrated over Qz")
ax.set_xlabel("Qx")
ax.set_ylabel("Qy")
add_window_lines(ax, x_low=qx_low, x_high=qx_high, y_low=qy_low, y_high=qy_high)
ax.add_patch(patches.Rectangle((qx_low, qy_low), qx_delta, qy_delta, linewidth=1, edgecolor="r", facecolor="none"))
fig.colorbar(im, ax=ax, label="Normalized intensity")
if xy_xlim is not None:
    ax.set_xlim(*xy_xlim)
if xy_ylim is not None:
    ax.set_ylim(*xy_ylim)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, f"{filebase}_xy_integrated_over_qz.tiff"), dpi=dpi, format="tiff")

# ================================
# Plot 1D integrations
# ================================

fig, axes = plt.subplots(1, 3, sharey=False, figsize=(15, 5))
axes[0].plot(qx, line_x_normalized, "b.", label="I(Qx), integrated over Qy-Qz window")
axes[1].plot(qy, line_y_normalized, "b.", label="I(Qy), integrated over Qx-Qz window")
axes[2].plot(qz, line_z_normalized, "b.", label="I(Qz), integrated over Qx-Qy window")

for ax, title, xlabel in zip(
    axes,
    ["Intensity vs. Qx", "Intensity vs. Qy", "Intensity vs. Qz"],
    ["Qx", "Qy", "Qz"],
):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized intensity")
    ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(output_dir, f"{filebase}_1d_integrations.tiff"), dpi=dpi, format="tiff")

# ================================
# Save integrated data
# ================================

line_x_error = np.full_like(line_x, fill_value=0.01, dtype=float)
line_y_error = np.full_like(line_y, fill_value=0.01, dtype=float)
line_z_error = np.full_like(line_z, fill_value=0.01, dtype=float)

if write_file:
    headerfmt = "#%11s%13s%13s\n"
    outfmt = " %10.6f %10.6e %10.6e"

    metadata = (
        f"# VTI file = {vti_input_file}\n"
        f"# No coordinate transform was applied. Original VTI qx/qy/qz axes were used.\n"
        f"# Qx_pos, Qx_delta = {qx_pos:8.4f}, {qx_delta:8.4f}\n"
        f"# Qy_pos, Qy_delta = {qy_pos:8.4f}, {qy_delta:8.4f}\n"
        f"# Qz_pos, Qz_delta = {qz_pos:8.4f}, {qz_delta:8.4f}\n"
        "#\n"
    )

    # Save text files inside the output folder
    with open(os.path.join(output_dir, f"{filebase}_qx.txt"), "w") as f:
        f.write(metadata)
        f.write(headerfmt % ("Qx", "I_sum", "Err"))
        np.savetxt(f, np.column_stack((qx, line_x, line_x_error)), fmt=outfmt)

    with open(os.path.join(output_dir, f"{filebase}_qy.txt"), "w") as f:
        f.write(metadata)
        f.write(headerfmt % ("Qy", "I_sum", "Err"))
        np.savetxt(f, np.column_stack((qy, line_y, line_y_error)), fmt=outfmt)

    with open(os.path.join(output_dir, f"{filebase}_qz.txt"), "w") as f:
        f.write(metadata)
        f.write(headerfmt % ("Qz", "I_sum", "Err"))
        np.savetxt(f, np.column_stack((qz, line_z, line_z_error)), fmt=outfmt)

    # Save .xye files beside the VTI file, matching the previous script behavior
    with open(os.path.join(data_dir, f"{filebase}_h.xye"), "w") as f:
        f.write(metadata)
        f.write(headerfmt % ("H", "Intensity", "Err"))
        np.savetxt(f, np.column_stack((qx, line_x, line_x_error)), fmt=outfmt)

    with open(os.path.join(data_dir, f"{filebase}_k.xye"), "w") as f:
        f.write(metadata)
        f.write(headerfmt % ("K", "Intensity", "Err"))
        np.savetxt(f, np.column_stack((qy, line_y, line_y_error)), fmt=outfmt)

    with open(os.path.join(data_dir, f"{filebase}_l.xye"), "w") as f:
        f.write(metadata)
        f.write(headerfmt % ("L", "Intensity", "Err"))
        np.savetxt(f, np.column_stack((qz, line_z, line_z_error)), fmt=outfmt)

if show_plots:
    plt.show()
else:
    plt.close("all")
