import os
import logging
from ase.io import write
from sklearn.model_selection import train_test_split

class MLFFTrain:
    """
    Handles the preprocessing of training data for different ML force field formats.
    Currently supports MACE.
    """

    def __init__(self, atoms_list, method="mace", output_dir="mlff_training_data"):
        """
        Parameters:
        - atoms_list (list of ASE Atoms): Training structures
        - method (str): The MLFF to format data for (e.g., 'mace', 'chgnet')
        - output_dir (str): Directory to write training data
        """
        self.atoms_list = atoms_list
        self.method = method.lower()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_training_data(self):
        """
        Dispatches the training data preparation to the appropriate method.
        """
        if self.method == "mace":
            return self._write_mace_xyz_split()
        else:
            raise NotImplementedError(f"MLFF method '{self.method}' is not implemented yet.")

    def _write_mace_xyz_split(self):
        """
        Splits and writes training/testing data for MACE.

        Returns:
        - dict: {'train_file': path, 'test_file': path}
        """
        train_data, test_data = train_test_split(self.atoms_list, test_size=0.1, random_state=42)
        logging.info(f"Total data: {len(self.atoms_list)}, Training data: {len(train_data)}, Testing data: {len(test_data)}")

        base = os.path.join(self.output_dir, "train_data")
        train_file = f"{base}_train.xyz"
        test_file = f"{base}_test.xyz"

        write(train_file, train_data, format="extxyz")
        write(test_file, test_data, format="extxyz")

        return {"train_file": train_file, "test_file": test_file}
