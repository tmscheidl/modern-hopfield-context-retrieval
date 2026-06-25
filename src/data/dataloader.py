import pytorch_lightning as pl
import numpy as np
import pickle
from torch.utils.data import DataLoader

# Fixed support set size, matching professor's fair few-shot protocol.
MAX_SUPPORT = 16  # padding/tensor size cap, independent of n_support


class FSMolDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None) -> None:
        self.databaseTraining = self._load_preprocessed_data(fold="training")
        self.databaseValidation = self._load_preprocessed_data(fold="validation")
        self.databaseTest = self._load_preprocessed_data(fold="test")

        self.databaseValidation = self._draw_fixed_support_and_query_set(
            self.databaseValidation
        )
        self.databaseTest = self._draw_fixed_support_and_query_set(self.databaseTest)

        self.trainingData = self._TrainingData(self.databaseTraining, self.config)
        self.validationData = self._EvalData(self.databaseValidation, self.config)
        self.testData = self._EvalData(self.databaseTest, self.config)

    def train_dataloader(self):
        return DataLoader(
            self.trainingData,
            batch_size=self.config.model.training.batch_size,
            shuffle=True,
            num_workers=self.config.system.ressources.num_workers_cpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validationData,
            batch_size=self.config.validation.batch_size,
            shuffle=False,
            num_workers=self.config.system.ressources.num_workers_cpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testData,
            batch_size=self.config.test.batch_size,
            num_workers=self.config.system.ressources.num_workers_cpu,
        )

    def _load_preprocessed_data(self, fold=["training", "validation", "test"]):
        if fold == "training":
            path = self.config.system.data.path + self.config.system.data.dir_training
        elif fold == "validation":
            path = self.config.system.data.path + self.config.system.data.dir_validation
        elif fold == "test":
            path = self.config.system.data.path + self.config.system.data.dir_test

        molIds = np.load(path + self.config.system.data.name_mol_ids)
        taskIds = np.load(path + self.config.system.data.name_target_ids)
        labels = np.load(path + self.config.system.data.name_labels).astype("float32")
        molInputs = np.load(path + self.config.system.data.name_mol_inputs).astype("float32")
        dictMolSmilesid = pickle.load(
            open(path + self.config.system.data.name_dict_mol_smiles_id, "rb")
        )
        dictTaskidActivemolecules = pickle.load(
            open(path + self.config.system.data.name_dict_target_id_activeMolecules, "rb")
        )
        dictTaskidInactivemolecules = pickle.load(
            open(path + self.config.system.data.name_dict_target_id_inactiveMolecules, "rb")
        )
        dictTasknamesId = pickle.load(
            open(path + self.config.system.data.name_dict_target_names_id, "rb")
        )

        dataDict = {
            "molIds": molIds,
            "taskIds": taskIds,
            "labels": labels,
            "molInputs": molInputs,
            "dictMolSmilesid": dictMolSmilesid,
            "dictTaskidActivemolecules": dictTaskidActivemolecules,
            "dictTaskidInactivemolecules": dictTaskidInactivemolecules,
            "dictTasknamesId": dictTasknamesId,
        }
        return dataDict

    def _draw_fixed_support_and_query_set(self, dataDict):
        n_support = int(self.config.supportSet.supportSetSize)

        dataDict["query_molIds"] = []
        dataDict["query_taskIds"] = []
        dataDict["query_labels"] = []
        dataDict["supportSetActives"] = {}
        dataDict["supportSetInactives"] = {}

        rng = np.random.default_rng(self.config.training.seed)

        for task_idx in list(dataDict["dictTaskidActivemolecules"]):
            active_mols_in_task = list(dataDict["dictTaskidActivemolecules"][task_idx])
            inactive_mols_in_task = list(dataDict["dictTaskidInactivemolecules"][task_idx])

            active_arr = np.array(active_mols_in_task)
            inactive_arr = np.array(inactive_mols_in_task)

            rng.shuffle(active_arr)
            rng.shuffle(inactive_arr)

            n_a = min(n_support, len(active_arr))
            n_i = min(n_support, len(inactive_arr))

            supportSetActivesIds = active_arr[:n_a]
            supportSetInactivesIds = inactive_arr[:n_i]

            queryActiveIds = active_arr[n_a:]
            queryInactiveIds = inactive_arr[n_i:]

            query_ids = np.concatenate([queryActiveIds, queryInactiveIds])
            query_labels = np.concatenate([
                np.ones(len(queryActiveIds)),
                np.zeros(len(queryInactiveIds)),
            ])

            dataDict["query_molIds"] += list(query_ids)
            dataDict["query_taskIds"] += list(np.repeat(task_idx, len(query_ids)))
            dataDict["query_labels"] += list(query_labels)
            dataDict["supportSetActives"][task_idx] = list(supportSetActivesIds)
            dataDict["supportSetInactives"][task_idx] = list(supportSetInactivesIds)

        return dataDict

    class _TrainingData:
        def __init__(self, database, config):
            self.database = database
            self.config = config
            self.len = len(self.database["molIds"])
            self.n_support = int(self.config.supportSet.supportSetSize)

        def __getitem__(self, index):
            molIdx = self.database["molIds"][index][0]
            queryMolecule = self.database["molInputs"][[molIdx], :]
            taskIdx = self.database["taskIds"][index][0]
            label = self.database["labels"][index]

            active_mols_in_task = self.database["dictTaskidActivemolecules"][taskIdx]
            inactive_mols_in_task = self.database["dictTaskidInactivemolecules"][taskIdx]

            if label == True:
                active_mols_in_task = [i for i in active_mols_in_task if i != molIdx]
            else:
                inactive_mols_in_task = [i for i in inactive_mols_in_task if i != molIdx]

            (
                supportSetActives,
                supportSetInactives,
                supportSetActivesSize,
                supportSetInactivesSize,
            ) = self.fixed_count_sampling_train(active_mols_in_task, inactive_mols_in_task)

            sample = {
                "queryMolecule": queryMolecule,
                "label": label,
                "supportSetActives": supportSetActives,
                "supportSetInactives": supportSetInactives,
                "supportSetActivesSize": supportSetActivesSize,
                "supportSetInactivesSize": supportSetInactivesSize,
                "taskIdx": taskIdx,
            }
            return sample

        def __len__(self):
            return self.len

        def fixed_count_sampling_train(self, active_mols_in_task, inactive_mols_in_task):
            active_arr = np.array(active_mols_in_task)
            inactive_arr = np.array(inactive_mols_in_task)

            if len(active_arr) > 0:
                np.random.shuffle(active_arr)
            if len(inactive_arr) > 0:
                np.random.shuffle(inactive_arr)

            n_a = min(self.n_support, len(active_arr))
            n_i = min(self.n_support, len(inactive_arr))

            supportSetActivesIds = active_arr[:n_a]
            supportSetInactivesIds = inactive_arr[:n_i]

            supportSetActives = self.database["molInputs"][supportSetActivesIds, :]
            supportSetInactives = self.database["molInputs"][supportSetInactivesIds, :]

            supportSetActives_size = supportSetActives.shape[0]
            supportSetInactives_size = supportSetInactives.shape[0]

            supportSetActives = supportSetActives[:MAX_SUPPORT, :]
            supportSetInactives = supportSetInactives[:MAX_SUPPORT, :]
            supportSetActives_size = min(supportSetActives_size, MAX_SUPPORT)
            supportSetInactives_size = min(supportSetInactives_size, MAX_SUPPORT)

            supportSetActives = np.pad(
                supportSetActives,
                ((0, MAX_SUPPORT - supportSetActives_size), (0, 0)),
                "constant",
                constant_values=0,
            )
            supportSetInactives = np.pad(
                supportSetInactives,
                ((0, MAX_SUPPORT - supportSetInactives_size), (0, 0)),
                "constant",
                constant_values=0,
            )

            return (
                supportSetActives,
                supportSetInactives,
                supportSetActives_size,
                supportSetInactives_size,
            )

    class _EvalData:
        def __init__(self, database, config):
            self.database = database
            self.config = config
            self.len = len(self.database["query_molIds"])

        def __getitem__(self, index):
            molIdx = self.database["query_molIds"][index]
            queryMolecule = self.database["molInputs"][[molIdx], :]
            taskIdx = self.database["query_taskIds"][index]
            label = self.database["query_labels"][index]

            supportSetActivesIndices = self.database["supportSetActives"][taskIdx]
            supportSetInactivesIndices = self.database["supportSetInactives"][taskIdx]

            supportSetActives = self.database["molInputs"][supportSetActivesIndices, :]
            supportSetInactives = self.database["molInputs"][supportSetInactivesIndices, :]

            supportSetActivesSize = supportSetActives.shape[0]
            supportSetInactivesSize = supportSetInactives.shape[0]

            supportSetActives = supportSetActives[:MAX_SUPPORT, :]
            supportSetInactives = supportSetInactives[:MAX_SUPPORT, :]
            supportSetActivesSize = min(supportSetActivesSize, MAX_SUPPORT)
            supportSetInactivesSize = min(supportSetInactivesSize, MAX_SUPPORT)

            supportSetActives = np.pad(
                supportSetActives,
                ((0, MAX_SUPPORT - supportSetActivesSize), (0, 0)),
                "constant",
                constant_values=0,
            )
            supportSetInactives = np.pad(
                supportSetInactives,
                ((0, MAX_SUPPORT - supportSetInactivesSize), (0, 0)),
                "constant",
                constant_values=0,
            )

            sample = {
                "queryMolecule": queryMolecule,
                "label": label,
                "supportSetActives": supportSetActives,
                "supportSetInactives": supportSetInactives,
                "supportSetActivesSize": supportSetActivesSize,
                "supportSetInactivesSize": supportSetInactivesSize,
                "taskIdx": taskIdx,
                "molIdx": molIdx,
            }
            return sample

        def __len__(self):
            return self.len