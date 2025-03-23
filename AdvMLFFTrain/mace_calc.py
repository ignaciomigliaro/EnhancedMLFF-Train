import os
import logging
from mace.calculators import MACECalculator
from ase import Atoms

class MaceCalc:
    """Handles loading MACE models and performing energy & force calculations."""

    def __init__(self, model_dir, device="cpu"):
        """
        Initializes MaceCalc with the model directory and device.

        Parameters:
        - model_dir (str): Path to the directory containing trained MACE models.
        - device (str): Device to run calculations ('cpu' or 'cuda').
        """
        self.model_dir = model_dir
        self.device = device
        self.models = self.load_models()
        self.num_models = len(self.models)  # Automatically determine number of models

    def load_models(self):
        """
        Loads MACE models from the specified directory.

        Returns:
        - list: List of model file paths.
        """
        models = []
        extension = ".model"  # MACE model files have .model extension
        for filename in os.listdir(self.model_dir):
            if filename.endswith(extension):
                model_path = os.path.join(self.model_dir, filename)
                models.append(model_path)
        logging.info(f"Successfully loaded {len(models)} MACE models.")
        return models

    def calculate_energy_forces(self, atoms_list):
        """
        Calculates total energy and forces for each configuration using all available MACE models.

        Parameters:
        - atoms_list (list): List of ASE Atoms objects.

        Returns:
        - tuple: (list of computed energies, list of computed forces)
        """
        if not self.models:
            logging.error("No MACE models loaded. Cannot perform calculations.")
            return None, None

        energies = []
        forces = []
        for atoms in atoms_list:
            model_energies = []
            model_forces = []
            for model_path in self.models:
                try:
                    calc = MACECalculator(model_paths=[model_path], device=self.device)
                    atoms.set_calculator(calc)

                    energy = atoms.get_total_energy()
                    force = atoms.get_forces()

                    model_energies.append(energy)
                    model_forces.append(force)
                except Exception as e:
                    logging.error(f"Error calculating energy/forces with model {model_path}: {e}")

            # Store results in atoms.info
            atoms.info["mace_energy"] = model_energies if self.num_models > 1 else model_energies[0]
            atoms.info["mace_forces"] = model_forces if self.num_models > 1 else model_forces[0]

            energies.append(atoms.info["mace_energy"])
            forces.append(atoms.info["mace_forces"])

        return atoms_list
