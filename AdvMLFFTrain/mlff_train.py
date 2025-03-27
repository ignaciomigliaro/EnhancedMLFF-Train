import os
import logging
from ase.io import write
from sklearn.model_selection import train_test_split
from AdvMLFFTrain.file_submit import Filesubmit  # adjust path if needed

class MLFFTrain:
    """
    Handles the preprocessing of training data for different ML force field formats.
    Currently supports MACE.
    """

    def __init__(self, atoms_list, method="mace", output_dir="models"):
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

    def prepare_and_submit_mlff(self):
        """
        Prepares training data and submits the MLFF training job.
        Dispatches logic based on selected MLFF method.
        """
        if self.method == "mace":
            train_file, test_file = self._write_mace_xyz_split()
            yaml_file = self.create_mace_yaml(train_file, test_file)
            self.submit_training_job(yaml_file)
        else:
            raise NotImplementedError(f"MLFF method '{self.method}' is not implemented yet.")

    def _write_mace_xyz_split(self):
        """
        Splits and writes training/testing data for MACE inside the output_dir.

        Returns:
        - dict: {'train_file': path, 'test_file': path}
        """
        train_data, test_data = train_test_split(self.atoms_list, test_size=0.1, random_state=42)
        logging.info(f"Total data: {len(self.atoms_list)}, Training data: {len(train_data)}, Testing data: {len(test_data)}")

        train_file = os.path.join(self.output_dir, "train.xyz")
        test_file = os.path.join(self.output_dir, "test.xyz")

        write(train_file, train_data, format="extxyz")
        write(test_file, test_data, format="extxyz")

        return {"train_file": train_file, "test_file": test_file}
    
    def create_mace_yaml(self, yaml_filename="mace_input.yaml", model_name="mace_model"):
        """
        Creates a new MACE YAML configuration in the template_dir,
        pointing to the train/test files in output_dir.

        Parameters:
        - yaml_filename (str): Name of the YAML file to create.
        - model_name (str): Optional model name used for directories.

        Returns:
        - str: Path to the created YAML config file.
        """
        yaml_path = os.path.join(self.template_dir, yaml_filename)
        
        config = {
            "name": model_name,
            "model_dir": self.output_dir,
            "log_dir": self.output_dir,
            "checkpoints_dir": self.output_dir,
            "train_file": os.path.join(self.output_dir, "train.xyz"),
            "test_file": os.path.join(self.output_dir, "test.xyz"),
            "energy_key": "dft_energy",
            "forces_key": "dft_forces",
            "swa": True,
            "max_num_epochs": 50,
            "batch_size": 10,
            "device": "cuda",
            "E0s": "average",
            "valid_fraction": 0.05,
            "save_cpu": True,
        }

        os.makedirs(self.template_dir, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f)

        logging.info(f"Created MACE config: {yaml_path}")
        return yaml_path

    def submit_training_job(self, yaml_path, slurm_name="mlff_train.slurm"):
        """
        Submit SLURM job using the given YAML and SLURM script from template_dir.
        """
        slurm_script_path = os.path.join(self.template_dir, slurm_name)

        if not os.path.exists(yaml_path):
            logging.error(f"YAML config not found: {yaml_path}")
            return None
        if not os.path.exists(slurm_script_path):
            logging.error(f"SLURM script not found: {slurm_script_path}")
            return None

        dest_slurm = os.path.join(self.output_dir, slurm_name)
        if not os.path.exists(dest_slurm):
            import shutil
            shutil.copy(slurm_script_path, dest_slurm)

        logging.info(f"Submitting SLURM job using: {dest_slurm}")
        try:
            result = subprocess.run(["sbatch", dest_slurm], check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split()[-1]
            logging.info(f"SLURM job submitted: {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to submit SLURM job: {e.stderr}")
            return None

    
