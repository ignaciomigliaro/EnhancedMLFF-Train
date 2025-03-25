import logging
import torch
from AdvMLFFTrain.mace_calc import MaceCalc
from AdvMLFFTrain.dft_input import DFTInputGenerator
from AdvMLFFTrain.utils import get_configurations, parse_orca_to_ase
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

class ActiveLearning:
    """Handles the active learning pipeline for MACE MLFF models."""

    def __init__(self, args):
        """
        Initializes the Active Learning pipeline with user-defined arguments.

        Parameters:
        - args (Namespace): Parsed command-line arguments.
        """
        self.args = args
        self.device = args.device
        self.calculator = args.calculator  # MACE or another DFT calculator
        self.output_dir = args.output_dir
        self.dft_software = args.dft_software
        self.eval_criteria = args.eval_criteria
        self.upper_threshold = args.threshold
        self.lower_threshold = args.lower_threshold
        self.use_cache = args.use_cache
        self.plot_std_dev = args.plot_std_dev
        self.atoms_list = []
        self.filtered_atoms_list = []

        os.makedirs(self.output_dir, exist_ok=True)

        logging.info(f"Using calculator: {self.calculator}")
        logging.info(f"Device selected: {self.device}")
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # **Initialize MACE calculator if selected**
        if self.calculator.lower() == "mace":
            self.mace_calc = MaceCalc(self.args.model_dir, self.device)

            # **Explicitly check models in model_dir**
            if not os.path.isdir(self.args.model_dir):
                raise ValueError(f"Model directory {self.args.model_dir} does not exist.")

            # **Ensure at least 3 models for active learning**
            if self.mace_calc.num_models < 3:
                raise ValueError(
                    f"Active Learning requires at least 3 MACE models, but only {self.mace_calc.num_models} were found in {self.args.model_dir}. "
                    f"Check if the correct models are present."
                )

            logging.info(f"Initialized MACE calculator with {self.mace_calc.num_models} models from {self.args.model_dir}.")

    def plot_std_dev_distribution(std_devs):
        """
        Plots the distribution of standard deviations using a histogram.

        Parameters:
        - std_devs (list): List of standard deviation values to plot.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(std_devs, bins=20, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Standard Deviations')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Frequency')
        plt.axvline(x=np.percentile(std_devs, 98), color='r', linestyle='--', label='98th Percentile')
        plt.legend()
        plt.grid(True)
        plt.show()

    def load_data(self):
        """Loads configurations using ASE-supported formats and initializes MACE models."""
        logging.info(f"Loading configurations from {self.args.filepath} using ASE.")

        # Load configurations using ASE-supported formats
        self.atoms_list = get_configurations(
            self.args.filepath, 
            self.args.stepsize, 
        )

        logging.info(f"Loaded {len(self.atoms_list)} configurations.")

    def calculate_energies_forces(self):
        """Assigns MACE calculators and computes energies & forces."""
        
        logging.info(f"Running MACE calculations on {len(self.atoms_list)} configurations.")

        # Compute energies and forces using MACE
        self.atoms_list = self.mace_calc.calculate_energy_forces(self.atoms_list)

        # Check if calculations were successful
        if not self.atoms_list or any("mace_energy" not in atoms.info or "mace_forces" not in atoms.info for atoms in self.atoms_list):
            logging.error("MACE calculations failed for some or all configurations.")
            return

        logging.info("Successfully computed energies and forces with MACE.")

    def calculate_std_dev(self, cache_file=None):
        """
        Calculate the standard deviation of energies and mean absolute deviation of forces
        for each configuration in the active learning set.

        Parameters:
        - cache_file (str, optional): Path to save computed energy values, forces, and deviations.

        Returns:
        - std_dev (list): Standard deviation of energies for each configuration.
        - mean_abs_deviation (list): Mean absolute deviation of forces for each atom in a configuration.
        """
        
        logging.info("Calculating standard deviations for energies and forces.")

        if not self.atoms_list:
            logging.error("No configurations available to compute standard deviation.")
            return None, None, None, None

        std_energy = []
        mean_abs_deviation = []

        progress = tqdm(total=len(self.atoms_list), desc="Processing Energies and Forces")

        for atoms in self.atoms_list:
            # Extract energy values for each model
            energy_values = atoms.info["mace_energy"]  # Should be a list of 3 values (one per model)
            std_energy.append(np.std(energy_values))

            # Extract forces (should be shape (3, N_atoms, 3))
            force_values = np.array(atoms.info["mace_forces"])  # Shape: (3, N_atoms, 3)

            # Compute mean absolute deviation of forces
            abs_deviation_values = []
            num_models = len(force_values)
            for i in range(num_models):
                for j in range(i + 1, num_models):
                    abs_deviation = np.abs(force_values[i] - force_values[j])
                    abs_deviation_values.append(np.mean(abs_deviation))

            mean_abs_deviation.append(np.mean(abs_deviation_values))  # Mean across all model comparisons
            
            progress.update(1)

        progress.close()

        self.std_dev = std_energy
        self.mean_abs_deviation = mean_abs_deviation

        logging.info("Standard deviations calculated for all configurations.")

        # Cache results if requested
        if cache_file:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'std_dev': std_energy,
                    'mean_abs_deviation': mean_abs_deviation
                }, f)

        return std_energy, mean_abs_deviation


    def filter_high_deviation_structures(self,percentile=90):
        """
        Filters structures based on the normalized standard deviation.
        Includes structures with normalized deviation within the specified threshold range.

        Parameters:
        - atoms_lists (list of list of ASE Atoms): List containing multiple atoms lists for each model.
        - energies (list of list of floats): List containing energies for each model.
        - std_dev (list of floats): Standard deviation values.
        - user_threshold (float, optional): User-defined upper threshold for filtering. If None, percentile-based threshold is used.
        - lower_threshold (float, optional): User-defined lower threshold for filtering. If None, no lower threshold is applied.
        - percentile (int): Percentile threshold for filtering if no user threshold is provided.

        Returns:
        - filtered_atoms_list (list of ASE Atoms): List of filtered structures.
        - filtered_std_dev (list of floats): List of standard deviation values corresponding to the filtered structures.
        """
        if self.eval_criteria == 'forces':
            std_dev = self.std_dev_forces
        if self.eval_criteria == 'energy':
            std_dev == self.std_dev

        std_dev_normalized = self.std_dev
        if self.upper_threshold and self.lower_threshold is not None:
            logging.info(f"User-defined upper threshold for filtering: {self.upper_threshold}")
        else:
            upper_threshold = np.percentile(std_dev, percentile)
            logging.info(f"Threshold for filtering (95th percentile): {percentile}")

        if lower_threshold is not None:
            lower_threshold = float(self.lower_threshold)
            logging.info(f"User-defined lower threshold for filtering: {lower_threshold}")
        else:
            lower_threshold = float('-inf')  # No lower threshold

        # Filter structures based on the chosen thresholds
        filtered_atoms_list = []
        filtered_std_dev = []

        for i, norm_dev in enumerate(std_dev):
            if lower_threshold <= norm_dev <= upper_threshold:  # Include structures within the threshold range
                filtered_atoms_list.append(self.atoms_lists[0][i])
                filtered_std_dev.append(norm_dev)
        logging.info(f"Number of structures within threshold range: {len(filtered_atoms_list)}")
        return filtered_atoms_list, filtered_std_dev

    def filter_structures(self, std_dev, mean_abs_deviation):
        """Filters structures based on energy or force standard deviation."""
        logging.info(f"Filtering structures based on {self.eval_criteria} standard deviation.")

        if self.eval_criteria == 'energy':
            self.filtered_atoms_list, _ = filter_high_deviation_structures(
                atoms_lists=self.atoms_list,
                std_dev=std_dev,
                user_threshold=self.threshold,
                lower_threshold=self.lower_threshold
            )
        elif self.eval_criteria == 'forces':
            self.filtered_atoms_list, _ = filter_high_deviation_structures(
                atoms_lists=self.atoms_list,
                std_dev=mean_abs_deviation,
                user_threshold=self.threshold,
                lower_threshold=self.lower_threshold
            )
        logging.info(f"Number of filtered structures: {len(self.filtered_atoms_list)}")

    def plot_std_dev_distribution(std_devs):
        """
        Plots the distribution of standard deviations using a histogram.

        Parameters:
        - std_devs (list): List of standard deviation values to plot.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(std_devs, bins=20, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Standard Deviations')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Frequency')
        plt.axvline(x=np.percentile(std_devs, 98), color='r', linestyle='--', label='98th Percentile')
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_dft_inputs(self):
        """Writes DFT input files for filtered structures."""
        if not self.dft_software:
            logging.warning("No DFT software specified. Skipping input file generation.")
            return

        dft_generator = dft_input(self.output_dir)

        for idx, atoms in enumerate(self.filtered_atoms_list):
            structure_output_dir = os.path.join(self.output_dir, f"structure_{idx}")
            os.makedirs(structure_output_dir, exist_ok=True)

            if self.dft_software.lower() == 'qe':
                dft_generator.write_qe_input(atoms, structure_output_dir)
            elif self.dft_software.lower() == 'orca':
                dft_generator.write_orca_input(atoms, template='orca_template.inp', filename="orca_input.inp")
            else:
                logging.error(f"Unsupported DFT software: {self.dft_software}")

    def run(self):
        """Executes the entire Active Learning pipeline."""
        self.load_data()
        self.calculate_energies_forces()
        std_dev, std_dev_forces = self.calculate_std_dev()
        self.filter_structures(std_dev, std_dev_forces)
        #TODO self.generate_dft_inputs()
        #TODO submit dft_inputs
        #TODO parse dft_inputs
        #TODO retrain mlff
        #TODO re-run
        logging.info("Active Learning process completed.")
