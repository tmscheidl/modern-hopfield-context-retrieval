import os
import sys
import copy
import pickle as pkl
from pickle import dump

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions.empirical_distribution import ECDF


# ============================================================
# CONFIG
# ============================================================

# Point to the FS-Mol repo folder (the one you extracted)
FS_MOL_CHECKOUT_PATH = r"C:\Users\tom39\Desktop\FS-Mol-main"

# Point to your actual data
FS_MOL_DATASET_PATH = r"C:\Users\tom39\Desktop\MHNfs\hopfield\data\fs-mol"

# Output directory
PREPROCESSED_PATH = r"C:\Users\tom39\Desktop\MHNfs\hopfield\data\preprocessed"

# Add FS-Mol to Python path so fs_mol can be imported
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)

# Debug mode
DEBUG_MODE = True
MAX_TASKS_DEBUG = 10


# ============================================================
# SETUP
# ============================================================

os.chdir(FS_MOL_CHECKOUT_PATH)
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)

from fs_mol.data import FSMolDataset, DataFold


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["training", "validation", "test"]:
    os.makedirs(os.path.join(PREPROCESSED_PATH, split), exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading FS-Mol dataset...")

dataset = FSMolDataset.from_directory(FS_MOL_DATASET_PATH)

print("Dataset loaded successfully.\n")


# ============================================================
# PREPROCESS FUNCTION
# ============================================================


def preprocess_split(
    datafold,
    split_name,
    descriptors_raw_forECDF=None,
    fit_scaler=False,
):
    """
    Preprocess one split of FS-Mol.

    Parameters
    ----------
    datafold : DataFold
        TRAIN / VALIDATION / TEST

    split_name : str
        training / validation / test

    descriptors_raw_forECDF : np.ndarray
        Raw training descriptors used for ECDF on val/test

    fit_scaler : bool
        True only for training split
    """

    print(f"==============================")
    print(f"Preprocessing split: {split_name}")
    print(f"==============================")

    output_dir = os.path.join(PREPROCESSED_PATH, split_name)

    # --------------------------------------------------------
    # Load task iterable
    # --------------------------------------------------------

    task_iterable = list(dataset.get_task_reading_iterable(datafold))

    if DEBUG_MODE:
        task_iterable = task_iterable[:MAX_TASKS_DEBUG]
        print(f"DEBUG MODE ACTIVE -> using {len(task_iterable)} tasks")

    # --------------------------------------------------------
    # Task names
    # --------------------------------------------------------

    tasks = [task.name for task in task_iterable]

    tasks_id_dict = {task_name: idx for idx, task_name in enumerate(tasks)}

    print(f"Number of tasks: {len(tasks)}")

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    mol_ids = []
    task_ids = []
    labels = []

    smiles_molId_dict = {}
    id_counter = 0

    fingerprints = {}
    descriptors = {}

    # --------------------------------------------------------
    # Read tasks
    # --------------------------------------------------------

    for task_idx, task in enumerate(task_iterable):

        print(f"Processing task {task_idx + 1}/{len(task_iterable)} -> {task.name}")

        for sample in task.samples:

            smiles = sample.smiles

            if smiles not in smiles_molId_dict:
                smiles_molId_dict[smiles] = id_counter
                id_counter += 1

            mol_id = smiles_molId_dict[smiles]

            mol_ids.append(mol_id)
            task_ids.append(tasks_id_dict[task.name])
            labels.append(sample.bool_label)

            if smiles not in fingerprints:
                fingerprints[smiles] = sample.fingerprint
                descriptors[smiles] = sample.descriptors

    print(f"Total molecules: {len(smiles_molId_dict)}")
    print(f"Total triplets: {len(mol_ids)}")

    # --------------------------------------------------------
    # Convert fingerprints/descriptors to numpy
    # --------------------------------------------------------

    fingerprints_temp = {}
    descriptors_temp = {}

    for key, value in fingerprints.items():
        fingerprints_temp[smiles_molId_dict[key]] = value

    for key, value in descriptors.items():
        descriptors_temp[smiles_molId_dict[key]] = value

    fingerprints_np = np.array(list(fingerprints_temp.values()))
    descriptors_np = np.array(list(descriptors_temp.values()))

    print(f"Fingerprints shape: {fingerprints_np.shape}")
    print(f"Descriptors shape: {descriptors_np.shape}")

    # --------------------------------------------------------
    # ECDF descriptors
    # --------------------------------------------------------

    if fit_scaler:
        descriptors_raw_forECDF = copy.deepcopy(descriptors_np)

        np.save(
            os.path.join(output_dir, "descriptors_raw_forECDF.npy"),
            descriptors_raw_forECDF,
        )

    descriptors_quantils = np.zeros_like(descriptors_np)

    print("Computing ECDF descriptor quantiles...")

    for column in range(descriptors_np.shape[1]):

        raw_values_ecdf = descriptors_raw_forECDF[:, column].reshape(-1)
        raw_values = descriptors_np[:, column].reshape(-1)

        ecdf = ECDF(raw_values_ecdf)
        quantils = ecdf(raw_values)

        descriptors_quantils[:, column] = quantils

    # --------------------------------------------------------
    # Combine fingerprints + descriptors
    # --------------------------------------------------------

    mol_inputs = np.hstack([fingerprints_np, descriptors_quantils])

    print(f"mol_inputs shape: {mol_inputs.shape}")

    # --------------------------------------------------------
    # Clean NaN / inf
    # --------------------------------------------------------

    mol_inputs[np.isnan(mol_inputs)] = 0
    mol_inputs[np.isinf(mol_inputs)] = 0

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(mol_inputs)

        dump(
            scaler,
            open(
                os.path.join(PREPROCESSED_PATH, "scaler_trainFitted.pkl"),
                "wb",
            ),
        )

        print("Training scaler fitted and saved.")

    else:
        scaler = pkl.load(
            open(
                os.path.join(PREPROCESSED_PATH, "scaler_trainFitted.pkl"),
                "rb",
            )
        )

        print("Training scaler loaded.")

    mol_inputs = scaler.transform(mol_inputs)

    # --------------------------------------------------------
    # Active/inactive dictionaries
    # --------------------------------------------------------

    triplett_ds = pd.DataFrame(
        {
            "mol": mol_ids,
            "task": task_ids,
            "labels": labels,
        }
    )

    task_actives = {}
    task_inactives = {}

    skipped_tasks = 0

    for task in np.unique(task_ids):

        subset_task = triplett_ds[triplett_ds["task"] == task]

        subset_actives = subset_task[subset_task["labels"] == True]
        subset_inactives = subset_task[subset_task["labels"] == False]

        set_actives = list(subset_actives["mol"])
        set_inactives = list(subset_inactives["mol"])

        if len(set_actives) == 0 or len(set_inactives) == 0:
            print(f"Skipping task {task} -> missing active/inactive samples")
            skipped_tasks += 1
            continue

        task_actives[task] = set_actives
        task_inactives[task] = set_inactives

    print(f"Skipped tasks: {skipped_tasks}")

    # --------------------------------------------------------
    # Save arrays
    # --------------------------------------------------------

    np.save(os.path.join(output_dir, "mol_inputs.npy"), mol_inputs)

    np.save(
        os.path.join(output_dir, "mol_ids.npy"),
        np.array(mol_ids).reshape(-1, 1),
    )

    np.save(
        os.path.join(output_dir, "task_ids.npy"),
        np.array(task_ids).reshape(-1, 1),
    )

    np.save(
        os.path.join(output_dir, "labels.npy"),
        np.array(labels).reshape(-1, 1),
    )

    # --------------------------------------------------------
    # Save dictionaries
    # --------------------------------------------------------

    dump(
        tasks_id_dict,
        open(os.path.join(output_dir, "dict_task_names_id.pkl"), "wb"),
    )

    dump(
        smiles_molId_dict,
        open(os.path.join(output_dir, "dict_mol_smiles_id.pkl"), "wb"),
    )

    dump(
        task_actives,
        open(
            os.path.join(output_dir, "dict_task_id_activeMolecules.pkl"),
            "wb",
        ),
    )

    dump(
        task_inactives,
        open(
            os.path.join(output_dir, "dict_task_id_inactiveMolecules.pkl"),
            "wb",
        ),
    )

    print(f"Finished preprocessing split: {split_name}\n")

    return descriptors_raw_forECDF

if __name__ == '__main__':

    # ============================================================
    # TRAIN
    # ============================================================

    train_descriptors_raw = preprocess_split(
        datafold=DataFold.TRAIN,
        split_name="training",
        fit_scaler=True,
    )

    # ============================================================
    # VALIDATION
    # ============================================================

    preprocess_split(
        datafold=DataFold.VALIDATION,
        split_name="validation",
        descriptors_raw_forECDF=train_descriptors_raw,
        fit_scaler=False,
    )

    # ============================================================
    # TEST
    # ============================================================

    preprocess_split(
        datafold=DataFold.TEST,
        split_name="test",
        descriptors_raw_forECDF=train_descriptors_raw,
        fit_scaler=False,
    )

    print("========================================")
    print("FS-Mol preprocessing completed.")
    print("========================================")