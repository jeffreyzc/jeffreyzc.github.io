import os
import csv
import math
from datetime import datetime

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

try:
    from ax.core.observation import ObservationFeatures
except Exception:
    ObservationFeatures = None


# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_FILE = "pld_experiment_log.csv"

OBJECTIVE_NAME = "roughness"

# If False, the script will stop when there is already a pending trial.
# This is safer for normal lab work.
# If True, you can generate multiple pending recipes before AFM is finished.
ALLOW_MULTIPLE_PENDING_TRIALS = False

MASTER_BOUNDS = {
    "temperature": [600.0, 750.0],        # °C
    "pressure": [1.0, 3.0],               # mTorr
    "laser_fluence": [4.0, 6.0],          # J/cm²
    "pulse_frequency": [2.0, 10.0],       # Hz
    "target_distance": [45.0, 60.0],      # mm
}

PARAM_INCREMENTS = {
    "temperature": 20.0,
    "pressure": 0.5,
    "laser_fluence": 0.5,
    "pulse_frequency": 1.0,
    "target_distance": 2.0,
}

PARAM_UNITS = {
    "temperature": "°C",
    "pressure": "mTorr",
    "laser_fluence": "J/cm²",
    "pulse_frequency": "Hz",
    "target_distance": "mm",
}

PARAM_DISPLAY_NAMES = {
    "temperature": "Temperature",
    "pressure": "Pressure",
    "laser_fluence": "Laser Fluence",
    "pulse_frequency": "Pulse Frequency",
    "target_distance": "Target Distance",
}

PARAM_LOOKUP = {
    1: "temperature",
    2: "pressure",
    3: "laser_fluence",
    4: "pulse_frequency",
    5: "target_distance",
}

CSV_COLUMNS = [
    "trial_id",
    "datetime",
    "temperature_C",
    "pressure_mTorr",
    "laser_fluence_J_cm2",
    "pulse_frequency_Hz",
    "target_distance_mm",
    "roughness_nm_rms",
    "status",
    "notes",
]


# =============================================================================
# HARDWARE GRID
# =============================================================================

def make_discrete_values(low, high, step):
    """
    Generate allowed hardware values.

    Example:
        pressure 1.0 to 3.0 step 0.5
        -> [1.0, 1.5, 2.0, 2.5, 3.0]

    Note:
        If the high value is not exactly on the grid, it is not included.

        Example:
        temperature 600 to 750 step 20
        -> 600, 620, 640, 660, 680, 700, 720, 740
        750 is not included because it is not reachable by 20 °C steps from 600.
    """
    values = []

    n_steps = int(math.floor((high - low) / step + 1e-12))

    for i in range(n_steps + 1):
        value = low + i * step
        if value <= high + 1e-12:
            values.append(round(value, 6))

    return [float(v) for v in values]


def get_allowed_values(param_name):
    low, high = MASTER_BOUNDS[param_name]
    step = PARAM_INCREMENTS[param_name]
    return make_discrete_values(low, high, step)


def validate_value(param_name, value):
    value = float(value)

    low, high = MASTER_BOUNDS[param_name]

    if value < low or value > high:
        raise ValueError(
            f"{param_name}={value} is outside bounds [{low}, {high}]"
        )

    allowed_values = get_allowed_values(param_name)

    if not any(abs(value - allowed) < 1e-8 for allowed in allowed_values):
        raise ValueError(
            f"{param_name}={value} is not on the allowed hardware grid. "
            f"Allowed values: {allowed_values}"
        )

    return value


def validate_recipe(recipe):
    for param_name in MASTER_BOUNDS:
        if param_name not in recipe:
            raise ValueError(f"Recipe missing parameter: {param_name}")

        validate_value(param_name, recipe[param_name])


# =============================================================================
# CSV HANDLING
# =============================================================================

def initialize_csv():
    """
    Create CSV file if it does not exist.
    """
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def read_csv_rows():
    initialize_csv()

    rows = []

    with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def write_csv_rows(rows):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for row in rows:
            clean_row = {col: row.get(col, "") for col in CSV_COLUMNS}
            writer.writerow(clean_row)


def get_next_lab_trial_id(rows):
    max_id = -1

    for row in rows:
        try:
            trial_id = int(row.get("trial_id", ""))
            max_id = max(max_id, trial_id)
        except Exception:
            pass

    return max_id + 1


def row_to_recipe(row):
    recipe = {
        "temperature": float(row["temperature_C"]),
        "pressure": float(row["pressure_mTorr"]),
        "laser_fluence": float(row["laser_fluence_J_cm2"]),
        "pulse_frequency": float(row["pulse_frequency_Hz"]),
        "target_distance": float(row["target_distance_mm"]),
    }

    validate_recipe(recipe)

    return recipe


def is_completed_row(row):
    status = row.get("status", "").strip().lower()
    roughness = row.get("roughness_nm_rms", "").strip()

    return status == "completed" and roughness != ""


def is_pending_row(row):
    status = row.get("status", "").strip().lower()
    roughness = row.get("roughness_nm_rms", "").strip()

    return status == "pending" or roughness == ""


def append_pending_trial_to_csv(recipe):
    rows = read_csv_rows()

    trial_id = get_next_lab_trial_id(rows)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = {
        "trial_id": trial_id,
        "datetime": now,
        "temperature_C": recipe["temperature"],
        "pressure_mTorr": recipe["pressure"],
        "laser_fluence_J_cm2": recipe["laser_fluence"],
        "pulse_frequency_Hz": recipe["pulse_frequency"],
        "target_distance_mm": recipe["target_distance"],
        "roughness_nm_rms": "",
        "status": "pending",
        "notes": "",
    }

    rows.append(new_row)
    write_csv_rows(rows)

    return trial_id


def print_pending_trials(rows):
    pending_rows = []

    for row in rows:
        try:
            if is_pending_row(row):
                recipe = row_to_recipe(row)
                pending_rows.append((row, recipe))
        except Exception:
            pass

    if not pending_rows:
        return []

    print("\n[!] Pending trials found in CSV:")
    for row, recipe in pending_rows:
        trial_id = row.get("trial_id", "?")

        print(f"\nLab Trial #{trial_id}")
        print_recipe(recipe, tuned_params=list(MASTER_BOUNDS.keys()))

    return pending_rows


# =============================================================================
# AX SETUP
# =============================================================================

def create_ax_client():
    """
    Create a fresh Ax optimizer from current script settings.
    No JSON is used.
    """
    ax_client = AxClient()

    ax_parameters = []

    for param_name in MASTER_BOUNDS:
        allowed_values = get_allowed_values(param_name)

        ax_parameters.append({
            "name": param_name,
            "type": "choice",
            "values": allowed_values,
            "value_type": "float",
            "is_ordered": True,
            "sort_values": True,
        })

    ax_client.create_experiment(
        name="pld_csv_only_optimization",
        parameters=ax_parameters,
        objectives={
            OBJECTIVE_NAME: ObjectiveProperties(minimize=True)
        },
    )

    return ax_client


def import_csv_history_into_ax(ax_client, rows):
    """
    Import CSV history into Ax.

    Completed rows:
        attached and completed with roughness.

    Pending rows:
        attached but not completed.
        This helps Ax avoid suggesting the exact same condition again.
    """
    imported_completed = 0
    imported_pending = 0
    skipped = 0

    for row in rows:
        try:
            recipe = row_to_recipe(row)

            if is_completed_row(row):
                roughness = float(row["roughness_nm_rms"])

                _, ax_trial_index = ax_client.attach_trial(parameters=recipe)

                ax_client.complete_trial(
                    trial_index=ax_trial_index,
                    raw_data={OBJECTIVE_NAME: roughness},
                )

                imported_completed += 1

            elif is_pending_row(row):
                _, ax_trial_index = ax_client.attach_trial(parameters=recipe)

                # Leave this trial running.
                imported_pending += 1

            else:
                skipped += 1

        except Exception as e:
            skipped += 1
            print(f"\n[skip] Could not import CSV row:")
            print(row)
            print(f"Reason: {e}")

    print("\nCSV history import:")
    print(f"  Completed trials imported: {imported_completed}")
    print(f"  Pending trials imported:   {imported_pending}")
    print(f"  Rows skipped:              {skipped}")

    return imported_completed, imported_pending, skipped


# =============================================================================
# USER INPUT
# =============================================================================

def safe_float_input(prompt, param_name=None):
    while True:
        text = input(prompt).strip()

        try:
            value = float(text)

            if param_name is not None:
                value = validate_value(param_name, value)

            return value

        except ValueError as e:
            print(f"\n[!] Invalid input: {e}\n")


def select_tuned_parameters():
    print("\n==================================================")
    print("       PLD ACTIVE EXPERIMENT INITIALIZATION       ")
    print("==================================================")
    print("Which parameters are you optimizing for THIS run?")
    print()

    for num, name in PARAM_LOOKUP.items():
        display = PARAM_DISPLAY_NAMES[name]
        unit = PARAM_UNITS[name]
        low, high = MASTER_BOUNDS[name]
        step = PARAM_INCREMENTS[name]
        allowed = get_allowed_values(name)

        print(f"  [{num}] {display}")
        print(f"      Bound: {low} to {high} {unit}, increment {step}")
        print(f"      Allowed: {allowed}")

    print("--------------------------------------------------")
    print("Enter option numbers separated by commas, e.g. 1,2,3")
    print("Or press Enter to optimize ALL parameters.")
    print("--------------------------------------------------")

    user_selection = input("Selection: ").strip()

    if user_selection == "":
        return list(MASTER_BOUNDS.keys())

    tuned_params = []

    for token in user_selection.split(","):
        try:
            num = int(token.strip())

            if num in PARAM_LOOKUP:
                tuned_params.append(PARAM_LOOKUP[num])

        except ValueError:
            pass

    tuned_params = list(dict.fromkeys(tuned_params))

    if not tuned_params:
        print("[!] No valid parameters selected. Optimizing all parameters.")
        tuned_params = list(MASTER_BOUNDS.keys())

    return tuned_params


def collect_fixed_parameters(tuned_params):
    fixed_parameters = {}

    for param_name in MASTER_BOUNDS:
        if param_name not in tuned_params:
            display = PARAM_DISPLAY_NAMES[param_name]
            unit = PARAM_UNITS[param_name]
            allowed = get_allowed_values(param_name)

            print()
            print(f"{display} is FIXED for this run.")
            print(f"Allowed values: {allowed}")

            prompt = f"Enter fixed value for {display} ({unit}): "

            fixed_parameters[param_name] = safe_float_input(
                prompt,
                param_name=param_name,
            )

    return fixed_parameters


def make_fixed_features(fixed_parameters):
    if not fixed_parameters:
        return None

    if ObservationFeatures is not None:
        return ObservationFeatures(parameters=fixed_parameters)

    return fixed_parameters


# =============================================================================
# DISPLAY
# =============================================================================

def print_recipe(recipe, tuned_params):
    validate_recipe(recipe)

    for param_name in MASTER_BOUNDS:
        display = PARAM_DISPLAY_NAMES[param_name]
        unit = PARAM_UNITS[param_name]
        value = float(recipe[param_name])
        status = "TUNED" if param_name in tuned_params else "FIXED"

        if param_name == "laser_fluence":
            print(f"  > {display:16s}: {value:7.2f} {unit:8s} ({status})")
        else:
            print(f"  > {display:16s}: {value:7.1f} {unit:8s} ({status})")


def print_best_so_far(ax_client):
    try:
        best_params, metrics = ax_client.get_best_parameters()
        validate_recipe(best_params)

        print("\n--- Best Historical Recommendation So Far ---")
        print_recipe(best_params, tuned_params=list(MASTER_BOUNDS.keys()))

        if metrics:
            print("\nModel-estimated objective information:")
            print(metrics)

    except Exception as e:
        print(f"\n[info] Best recommendation not available yet: {e}")


def print_startup_message():
    print("\n==================================================")
    print("        PLD BAYESIAN OPTIMIZATION SCRIPT          ")
    print("                  CSV-ONLY MODE                   ")
    print("==================================================")
    print("This script does not use JSON.")
    print()
    print("Workflow:")
    print("  1. Read completed trials from CSV")
    print("  2. Rebuild optimizer")
    print("  3. Suggest one new PLD recipe")
    print("  4. Append recipe to CSV as pending")
    print("  5. Exit")
    print()
    print("After AFM:")
    print("  Fill roughness_nm_rms")
    print("  Change status from pending to completed")
    print("==================================================")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_startup_message()

    initialize_csv()
    rows = read_csv_rows()

    pending_rows = print_pending_trials(rows)

    if pending_rows and not ALLOW_MULTIPLE_PENDING_TRIALS:
        print("\n[STOP]")
        print("There is already at least one pending trial in the CSV.")
        print("Run the pending recipe first, then after AFM fill:")
        print("  roughness_nm_rms")
        print("  status = completed")
        print()
        print("If you intentionally want multiple pending trials, set:")
        print("  ALLOW_MULTIPLE_PENDING_TRIALS = True")
        return

    ax_client = create_ax_client()
    imported_completed, imported_pending, skipped = import_csv_history_into_ax(
        ax_client,
        rows,
    )

    print_best_so_far(ax_client)

    tuned_params = select_tuned_parameters()
    fixed_parameters = collect_fixed_parameters(tuned_params)

    print("\nSelected tuned parameters:")
    for param_name in tuned_params:
        print(f"  - {PARAM_DISPLAY_NAMES[param_name]}")

    if fixed_parameters:
        print("\nFixed parameters:")
        for param_name, value in fixed_parameters.items():
            print(
                f"  - {PARAM_DISPLAY_NAMES[param_name]}: "
                f"{value} {PARAM_UNITS[param_name]}"
            )

    try:
        fixed_features = make_fixed_features(fixed_parameters)

        if fixed_features is not None:
            suggested_params, ax_trial_index = ax_client.get_next_trial(
                fixed_features=fixed_features
            )
        else:
            suggested_params, ax_trial_index = ax_client.get_next_trial()

    except Exception as e:
        print("\n[!] Error generating new trial:")
        print(e)
        print()
        print("Possible causes:")
        print("  - Too many pending trials.")
        print("  - Fixed values make the search impossible.")
        print("  - CSV contains invalid rows.")
        return

    recipe = {}

    for param_name in MASTER_BOUNDS:
        recipe[param_name] = float(suggested_params[param_name])

    try:
        validate_recipe(recipe)

    except Exception as e:
        print("\n[CRITICAL SAFETY STOP]")
        print("Suggested recipe is outside the allowed PLD grid.")
        print(e)
        return

    lab_trial_id = append_pending_trial_to_csv(recipe)

    print(f"\n--- New Recommended PLD Recipe: Lab Trial #{lab_trial_id} ---")
    print_recipe(recipe, tuned_params=tuned_params)

    print("\n[✓] New pending trial appended to CSV.")
    print(f"CSV file: {CSV_FILE}")

    print("\nAfter deposition and AFM, update this CSV row:")
    print("  roughness_nm_rms = your measured AFM RMS roughness")
    print("  status = completed")

    print("\nExample:")
    print("  roughness_nm_rms = 2.85")
    print("  status = completed")


if __name__ == "__main__":
    main()
