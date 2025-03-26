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
        self.template_dir = args.template_dir if self.dft_software.lower() == "orca" else None
        self.eval_criteria = args.eval_criteria
        self.upper_threshold = args.upper_threshold
        self.lower_threshold = args.lower_threshold
        self.use_cache = args.use_cache
        self.plot_std_dev = args.plot_std_dev
        self.sample_percentage = args.sample_percentage

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
        sampled_atoms,remaining_atoms = get_configurations(
            self.args.filepath, 
            self.args.sample_percentage, 
        )

        logging.info(f"Loaded {len(sampled_atoms)} configurations.")
        return sampled_atoms, remaining_atoms

    def calculate_energies_forces(self,sampled_atoms):
        """Assigns MACE calculators and computes energies & forces."""
        
        logging.info(f"Running MACE calculations on {len(sampled_atoms)} configurations.")

        # Compute energies and forces using MACE
        sampled_atoms = self.mace_calc.calculate_energy_forces(sampled_atoms)

        # Check if calculations were successful
        if not sampled_atoms or any("mace_energy" not in atoms.info or "mace_forces" not in atoms.info for atoms in sampled_atoms):
            logging.error("MACE calculations failed for some or all configurations.")
            return

        logging.info("Successfully computed energies and forces with MACE.")

        return sampled_atoms

    def calculate_std_dev(self, sampled_atoms):
        """
        Calculate the standard deviation of energies and forces for each atomic configuration
        in the active learning set (Query by Committee).

        Parameters:
        - cache_file (str, optional): Path to save computed energy values, forces, and deviations.

        Returns:
        - std_energy (list): Standard deviation of energies per configuration.
        - std_dev_forces (list): Standard deviation of forces per atom in each configuration.
        - energy_values (list): Computed energy values.
        - force_values (list): Computed force values.
        """
        
        logging.info("Calculating standard deviations for energies and forces.")

        if not sampled_atoms:
            logging.error("No configurations available to compute standard deviation.")
            return None, None, None, None

        std_energy = []
        std_dev_forces = []

        progress = tqdm(total=len(sampled_atoms), desc="Processing Energies and Forces")

        for atoms in sampled_atoms:
            # Extract energy values from different models
            energy_values = atoms.info["mace_energy"]  # Should be a list of 3 values (one per model)
            std_energy.append(np.std(energy_values))  # Compute standard deviation of energies

            # Extract forces from different models
            force_values = np.array(atoms.info["mace_forces"])  # Shape: (3, N_atoms, 3)

            # Compute standard deviation of forces across models for each atom
            std_dev_atom_forces = np.std(force_values, axis=0)  # Shape: (N_atoms, 3)
            std_dev_forces.append(np.mean(std_dev_atom_forces))  # Mean over all atoms and directions

            progress.update(1)

        progress.close()

        #self.std_dev = std_energy
        #self.std_dev_forces = std_dev_forces  # This is now correctly computed

        logging.info("Standard deviations calculated for all configurations.")

        return std_energy, std_dev_forces

    def filter_high_deviation_structures(self,std_dev,std_dev_forces,sampled_atoms,percentile=90):
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
            std_dev = std_dev_forces
        if self.eval_criteria == 'energy':
            std_dev == std_dev
        
        if self.upper_threshold and self.lower_threshold is not None:
            logging.info(f"User-defined upper threshold for filtering: {self.upper_threshold}")
        else:
            upper_threshold = np.percentile(std_dev, percentile)
            logging.info(f"Threshold for filtering (95th percentile): {percentile}")

        if self.lower_threshold is not None:
            lower_threshold = float(self.lower_threshold)
            logging.info(f"User-defined lower threshold for filtering: {lower_threshold}")
        else:
            lower_threshold = float('-inf')  # No lower threshold

        # Filter structures based on the chosen thresholds
        filtered_atoms_list = []
        filtered_std_dev = []

        for i, norm_dev in enumerate(std_dev):
            if self.lower_threshold <= norm_dev <= self.upper_threshold:  # Include structures within the threshold range
                filtered_atoms_list.append(sampled_atoms[i])
                filtered_std_dev.append(norm_dev)
        logging.info(f"Number of structures within threshold range: {len(filtered_atoms_list)}")
        return filtered_atoms_list

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

    def generate_dft_inputs(self, atoms_list):
        """
        Generate DFT input files for ORCA or QE based on `self.dft_software`.

        Parameters:
        - atoms_list (list): List of ASE Atoms objects.

        Returns:
        - input_files (list): List of generated input file paths.
        """
        dft_input = DFTInputGenerator(
        output_dir="DFT_inputs", 
        dft_software=self.dft_software, 
        template_dir=self.template_dir  # Pass template directory
    )
        return dft_input.generate_dft_inputs(atoms_list)

    
    def run(self):
        """Executes the entire Active Learning pipeline."""
        sampled_atoms, remaining_atoms = self.load_data()
        sampled_atoms = self.calculate_energies_forces(sampled_atoms)
        std_dev, std_dev_forces = self.calculate_std_dev(sampled_atoms)
        filtered_atoms_list = self.filter_high_deviation_structures(std_dev,std_dev_forces,sampled_atoms)
        self.generate_dft_inputs(filtered_atoms_list)
        #TODO submit dft_inputs
        #TODO parse dft_inputs
        #TODO retrain mlff
        #TODO re-run
        logging.info("Active Learning process completed.")
