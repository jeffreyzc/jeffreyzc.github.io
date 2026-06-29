import os
import glob
import re
import numpy as np

import matplotlib.pyplot as plt
import tifffile as tiff

# --- EXPERIMENTAL CONSTANTS ---
PIXEL_SIZE_MM = 0.055      # Timepix detector
SDD_MM = 840.0             # 0.84 meters
WAVELENGTH = 1.393         # 8.9 keV
DETECTOR_CENTER_2THETA = 37.76 # From Nu motor in SPEC file

def get_master_image(folder_path):
    file_pattern = os.path.join(folder_path, "*.tif")
    file_list = sorted(glob.glob(file_pattern))
    if not file_list: return None
    first_img = tiff.imread(file_list[0])
    master_img = np.zeros(first_img.shape, dtype=np.float32)
    for file in file_list: master_img += np.maximum(tiff.imread(file), 0)
    return master_img / len(file_list)

def calibrate_x_axis(pixel_axis, reference_pixel):
    distance_mm = (pixel_axis - reference_pixel) * PIXEL_SIZE_MM
    delta_2theta_rad = np.arctan(distance_mm / SDD_MM)
    absolute_2theta = DETECTOR_CENTER_2THETA + np.degrees(delta_2theta_rad)
    theta_rad = np.radians(absolute_2theta / 2.0)
    q_space = (4 * np.pi / WAVELENGTH) * np.sin(theta_rad)
    return absolute_2theta, q_space

def normalize_0_to_1(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def calculate_fwhm(x_axis, y_axis):
    peak_idx = np.argmax(y_axis)
    peak_pos = x_axis[peak_idx]
    above_half = np.where(y_axis >= 0.5)[0]
    if len(above_half) == 0: return peak_pos, 0.0 
    idx_left, idx_right = above_half[0], above_half[-1]
    
    if idx_left > 0:
        x0, y0 = x_axis[idx_left-1], y_axis[idx_left-1]
        x1, y1 = x_axis[idx_left], y_axis[idx_left]
        x_left = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
    else: x_left = x_axis[0]
        
    if idx_right < len(x_axis) - 1:
        x2, y2 = x_axis[idx_right], y_axis[idx_right]
        x3, y3 = x_axis[idx_right+1], y_axis[idx_right+1]
        x_right = x2 + (0.5 - y2) * (x3 - x2) / (y3 - y2)
    else: x_right = x_axis[-1]
    return peak_pos, x_right - x_left

# ==========================================
# 1. SETUP FOLDERS
# ==========================================

folders = [
    r"file path 1",
    r"file path 2", 
    r"file path 3",
    r"file path 4"
]

master_images, folder_names_clean = [], []
for folder in folders:
    img = get_master_image(folder)
    if img is not None:
        master_images.append(img)
        folder_names_clean.append(os.path.basename(os.path.normpath(folder)))

if len(master_images) >= 2:
    ref_img = master_images[0]
    _, x_center_ref = np.unravel_index(np.argmax(ref_img), ref_img.shape)
    half_width, swath_thickness = 100, 15
    
    x_range = slice(max(0, x_center_ref - half_width), min(ref_img.shape[1], x_center_ref + half_width))
    pixel_axis = np.arange(x_range.start, x_range.stop)
    two_theta_axis, q_axis = calibrate_x_axis(pixel_axis, x_center_ref)

    # ==========================================
    # 2. TOGGLES & DATA EXTRACTION
    # ==========================================
    PLOT_UNIT = "2Theta"       # Options: "q" or "2Theta"
    PLOT_SCALE = "linear"      # Options: "log" or "linear"
    
    # ---> SET YOUR X-AXIS LIMITS HERE <---
    Q_LIMITS = (2.91, 2.93)
    THETA_LIMITS = (37.68, 37.86)
    # -------------------------------------
    
    if PLOT_UNIT == "q":
        x_data, x_label, unit_str = q_axis, r"Scattering Vector, $q \ (\AA^{-1})$", r"$\AA^{-1}$"
        xl1, xl2 = Q_LIMITS
        header_str = "q(A^-1)\tNormalized_Intensity"
    else:
        x_data, x_label, unit_str = two_theta_axis, r"2Theta (°)", "Deg"
        xl1, xl2 = THETA_LIMITS
        header_str = "2Theta(deg)\tNormalized_Intensity"

    plot_colors = [
        '#000000',  # Black (Added)
        '#E41A1C',  # Red
        '#377EB8',  # Steel Blue
        '#FFC20A',  # Gold/Yellow
        '#4DAF4A',  # Green
        '#7570B3',  # Purple
        '#999999',  # Gray
        '#FB8072',  # Coral/Salmon
        '#80B1D3'   # Light Blue
    ]
    times, fwhm_list, q_peak_list = [], [], []

    # --- PLOT 1: DIFFRACTION OVERLAY ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    for i, img in enumerate(master_images):
        folder_name = folder_names_clean[i]
        safe_color = plot_colors[i % len(plot_colors)]
        
        # Regex time extraction
        time_match = re.search(r"(-?\d+\.?\d*)\s*ns", folder_name.replace(" .", "."))
        times.append(float(time_match.group(1)) if time_match else i)

        search_area = img[:, x_range]
        y_center_local = np.unravel_index(np.argmax(search_area), search_area.shape)[0]
        y_range_local = slice(max(0, y_center_local - swath_thickness), min(img.shape[0], y_center_local + swath_thickness))
        
        raw_1d_slice = np.mean(img[y_range_local, x_range], axis=0)
        norm_1d_slice = normalize_0_to_1(raw_1d_slice)
        
        peak_pos, fwhm_val = calculate_fwhm(x_data, norm_1d_slice)
        peak_q, _ = calculate_fwhm(q_axis, norm_1d_slice)
        
        fwhm_list.append(fwhm_val)
        q_peak_list.append(peak_q)
        
        #legend_label = f"{folder_name}\n  Peak: {peak_pos:.5f} {unit_str}\n  FWHM: {fwhm_val:.5f} {unit_str}"
        legend_label = folder_name
        print(f"[{folder_name}] Peak: {peak_pos:.5f} {unit_str} | FWHM: {fwhm_val:.5f} {unit_str}")
        ax1.plot(x_data, norm_1d_slice, color=safe_color, linewidth=3, label=legend_label)
        
        # Export
        np.savetxt(f"{folder_name}_1D_profile.txt", np.column_stack((x_data, norm_1d_slice)), fmt="%.6f", delimiter="\t", header=header_str)

    ax1.set_xlabel(x_label, fontsize=13)
    ax1.set_ylabel(f"{'Log ' if PLOT_SCALE == 'log' else ''}Normalized Intensity (a.u.)", fontsize=13)
    ax1.set_xlim(xl1, xl2)
    ax1.set_yscale(PLOT_SCALE)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right', fontsize=14, framealpha=0.9)
    plt.title("Plot 1: Normalized Overlay of (002) Profiles", fontweight='bold', fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 3. CALCULATE DYNAMICS ARRAYS
    # ==========================================
    times, fwhm_list, q_peak_list = np.array(times), np.array(fwhm_list), np.array(q_peak_list)
    sort_idx = np.argsort(times)
    times, fwhm_list, q_peak_list = times[sort_idx], fwhm_list[sort_idx], q_peak_list[sort_idx]

    fwhm_pct = ((fwhm_list - fwhm_list[0]) / fwhm_list[0]) * 100.0
    bulk_strain_pct = (-(q_peak_list - q_peak_list[0]) / q_peak_list[0]) * 100.0

    # --- PLOT 2: ABSOLUTE FWHM ---
    plt.figure(figsize=(8, 5))
    plt.plot(times, fwhm_list, 'rs-', linewidth=2, markersize=8)
    plt.xlabel("Time (ns)", fontsize=13)
    plt.ylabel(f"FWHM ({unit_str})", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.title("Plot 2: Absolute FWHM vs Time", fontweight='bold', fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()

    # --- PLOT 3: FWHM PERCENTAGE CHANGE ---
    plt.figure(figsize=(8, 5))
    plt.plot(times, fwhm_pct, 'ko-', linewidth=2, markersize=8)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.6)
    plt.xlabel("Time (ns)", fontsize=13)
    plt.ylabel(r"$\Delta$ FWHM (%)", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.title("Plot 3: FWHM Change (%) vs Time", fontweight='bold', fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()

    # --- PLOT 4: DUAL AXIS (FWHM & STRAIN) ---
    fig4, ax_left = plt.subplots(figsize=(9, 5))
    ax_right = ax_left.twinx() # Create the second Y-axis

    # Left Axis (FWHM)
    line1 = ax_left.plot(times, fwhm_pct, 'ko', markersize=8, label="(002) $\Delta$ FWHM (%)")
    ax_left.set_xlabel("Time (ns)", fontsize=13)
    ax_left.set_ylabel("$\Delta$ FWHM (%)", fontsize=13, color='black')
    ax_left.tick_params(axis='y', labelcolor='black')
    
    # Right Axis (Strain)
    line2 = ax_right.plot(times, bulk_strain_pct, 'r-', linewidth=2.5, label="Bulk Strain (%)")
    ax_right.set_ylabel("Bulk Strain (%)", fontsize=13, color='red')
    ax_right.tick_params(axis='y', labelcolor='red')

    # Add baseline and combine legends
    ax_left.axhline(0, color='gray', linestyle='--', alpha=0.5)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc='upper left', fontsize=11)

    ax_left.grid(True, linestyle='--', alpha=0.4)
    plt.title("Plot 4: FWHM & Bulk Strain Dynamics (Dual Axis)", fontweight='bold', fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()

    # --- PLOT 5: STRAIN PERCENTAGE ONLY ---
    plt.figure(figsize=(8, 5))
    plt.plot(times, bulk_strain_pct, 'rs-', linewidth=2.5, markersize=8, label="Bulk Strain")
    plt.axhline(0, color='gray', linestyle='--', alpha=0.6)
    plt.xlabel("Time (ns)", fontsize=13)
    plt.ylabel("Bulk Strain (%)", fontsize=13, color='black')
    plt.tick_params(axis='y', labelcolor='black')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.title("Plot 5: Lattice Strain (%) vs Time", fontweight='bold', fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()
