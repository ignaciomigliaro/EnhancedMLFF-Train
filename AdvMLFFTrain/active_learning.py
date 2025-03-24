import logging
import torch
from AdvMLFFTrain.mace_calc import MaceCalc
from AdvMLFFTrain.dft_input import DFTInputGenerator
from AdvMLFFTrain.utils import get_configurations
import os
import matplotlib.pyplot as plt
import numpy as np

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
        self.calculator = args.calculator
        self.output_dir = args.output_dir
        self.dft_software = args.dft_software
        self.eval_criteria = args.eval_criteria
        self.threshold = args.threshold
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
        """Loads configurations and models."""
        self.atoms_list = get_configurations(self.args.filepath, self.args.stepsize)
        logging.info(f"Loaded {len(self.atoms_list)} configurations.")

        self.mace_calc = MaceCalc(self.args.model_dir, self.device)
        logging.info(f"Models loaded: {len(self.mace_calc.models)}")
    
    def calculate_std_dev(atoms_list, cache_file=None):
        """
        Calculate the standard deviation of energies and forces across different MACE models.
        
        Parameters:
        - atoms_list (list): List of ASE Atoms objects with precomputed energy and force values in atoms.info.
        - cache_file (str, optional): Path to a file where the computed data will be cached.

        Returns:
        - std_dev (list): Standard deviation of energies for each configuration.
        - std_dev_forces (list): Standard deviation of forces for each configuration.
        - energy_values (list): List of energy values for each configuration.
        - force_values (list): List of force values for each configuration.
        """
        if not atoms_list:
            logging.error("Empty atoms_list provided. Cannot calculate standard deviations.")
            return None, None, None, None

        num_configs = len(atoms_list)
        energies = []
        forces = []

        progress = tqdm(total=num_configs, desc="Processing Energies and Forces")

        # Collect energies and forces from atoms.info
        for atoms in atoms_list:
            try:
                energy = atoms.info.get("mace_energy", None)
                force = atoms.info.get("mace_forces", None)

                if energy is not None and force is not None:
                    energies.append(energy)
                    forces.append(np.array(force).flatten())
                else:
                    logging.warning(f"Missing energy or force data in atoms.info for {atoms}. Skipping entry.")

            except Exception as e:
                logging.error(f"Error retrieving energy/forces from atoms.info: {e}")
            
            progress.update(1)

        # Convert to NumPy arrays
        energies_array = np.array(energies)
        forces_array = np.array(forces)

        # Compute standard deviation of energies
        std_dev = np.std(energies_array, axis=0).tolist()  # Standard deviation across models

        # Compute standard deviation of forces
        std_dev_forces = np.std(forces_array, axis=0).tolist()  # Standard deviation across models

        # Cache results if requested
        if cache_file:
            data_to_save = {
                'energy_values': energies_array.tolist(),
                'force_values': forces_array.tolist(),
                'std_dev': std_dev,
                'std_dev_forces': std_dev_forces
            }
            save_to_cpu_pickle(data_to_save, cache_file)  # Save results

        return std_dev, std_dev_forces, energies_array.tolist(), forces_array.tolist()
    
    def calculate_energies_forces(self):
        """Assigns calculators and computes energies & forces."""
        logging.info(f"Running calculations on {len(self.atoms_list)} configurations.")
        _, mean_abs_deviation, std_dev, _ = calculate_std_dev(self.atoms_list, cache_file=self.use_cache)
        logging.info(f"Standard deviations calculated for {len(std_dev)} atoms.")

        if self.plot_std_dev:
            plot_std_dev_distribution(std_dev)

        return std_dev, mean_abs_deviation

    def filter_high_deviation_structures(atoms_lists, std_dev, user_threshold=None, lower_threshold=None, percentile=90):
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
        # Compute the normalized standard deviation
        std_dev_normalized = std_dev
        if user_threshold is not None:
            upper_threshold = float(user_threshold)
            logging.info(f"User-defined upper threshold for filtering: {upper_threshold}")
        else:
            upper_threshold = np.percentile(std_dev_normalized, percentile)
            logging.info(f"Threshold for filtering (95th percentile): {upper_threshold}")

        if lower_threshold is not None:
            lower_threshold = float(lower_threshold)
            logging.info(f"User-defined lower threshold for filtering: {lower_threshold}")
        else:
            lower_threshold = float('-inf')  # No lower threshold

        # Filter structures based on the chosen thresholds
        filtered_atoms_list = []
        filtered_std_dev = []
        for i, norm_dev in enumerate(std_dev_normalized):
            if lower_threshold <= norm_dev <= upper_threshold:  # Include structures within the threshold range
                filtered_atoms_list.append(atoms_lists[0][i])
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
        atoms_list = self.calculate_energies_forces()
        std_dev, std_dev_forces =self.calculate_std_dev(atoms_list)
        self.filter_structures(std_dev, std_dev_forces)
        #TODO self.generate_dft_inputs()
        #TODO submit dft_inputs
        #TODO parse dft_inputs
        #TODO retrain mlff
        #TODO re-run
        logging.info("Active Learning process completed.")
