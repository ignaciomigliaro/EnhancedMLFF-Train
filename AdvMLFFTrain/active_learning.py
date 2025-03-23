MaceCalcimport os
import logging
import torch
from AdvMLFFTrain.config_loader import get_configurations
from AdvMLFFTrain.mace_calc import MaceCalc
from AdvMLFFTrain.dft_input import ActiveLearningDFT

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

    def load_data(self):
        """Loads configurations and models."""
        self.atoms_list = get_configurations(self.args.filepath, self.args.stepsize)
        logging.info(f"Loaded {len(self.atoms_list)} configurations.")

        self.mace_calc = MaceCalc(self.args.model_dir, self.device)
        logging.info(f"Models loaded: {len(self.mace_calc.models)}")

    def calculate_energies_forces(self):
        """Assigns calculators and computes energies & forces."""
        logging.info(f"Running calculations on {len(self.atoms_list)} configurations.")
        _, mean_abs_deviation, std_dev, _ = calculate_std_dev(self.atoms_list, cache_file=self.use_cache)
        logging.info(f"Standard deviations calculated for {len(std_dev)} atoms.")

        if self.plot_std_dev:
            plot_std_dev_distribution(std_dev)

        return std_dev, mean_abs_deviation

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

    def generate_dft_inputs(self):
        """Writes DFT input files for filtered structures."""
        if not self.dft_software:
            logging.warning("No DFT software specified. Skipping input file generation.")
            return

        dft_generator = ActiveLearningDFT(self.output_dir)

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
        std_dev, mean_abs_deviation = self.calculate_energies_forces()
        self.filter_structures(std_dev, mean_abs_deviation)
        self.generate_dft_inputs()
        logging.info("Active Learning process completed.")
