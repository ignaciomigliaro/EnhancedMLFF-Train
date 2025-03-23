import os
import logging
import subprocess
import yaml
import time
from pathlib import Path

class MaceTrain:
    """Handles training and monitoring of MACE MLFF models on SLURM."""

    def __init__(self, yaml_config_path):
        """
        Initialize MaceTrain with the YAML configuration file path.
        """
        self.yaml_config_path = yaml_config_path
        self.yaml_config = self.read_yaml_file()
        self.slurm_job_id = None  # Stores the submitted SLURM job ID

    def read_yaml_file(self):
        """
        Reads the YAML configuration file and returns the parsed configuration.
        """
        if not os.path.exists(self.yaml_config_path):
            logging.error(f"Error: YAML configuration file '{self.yaml_config_path}' not found.")
            return None
        try:
            with open(self.yaml_config_path, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)
            logging.info(f"YAML configuration loaded successfully from: {self.yaml_config_path}")
            return config
        except Exception as e:
            logging.error(f"Error reading YAML configuration file: {e}")
            return None

    def submit_training(self):
        """
        Submits a MACE training job to SLURM using the configuration.
        """
        if self.yaml_config is None:
            logging.error("Error: YAML configuration is missing.")
            return None

        slurm_script = self.yaml_config.get("slurm_script", "")
        if not slurm_script:
            logging.error("Error: SLURM script path is missing in YAML configuration.")
            return None

        try:
            # Submit the SLURM job
            command = f"sbatch {slurm_script}"
            output = subprocess.check_output(command, shell=True, text=True).strip()
            logging.info(f"SLURM job submitted: {output}")

            # Extract job ID from SLURM output
            if "Submitted batch job" in output:
                self.slurm_job_id = output.split()[-1]  # Extract job ID
                logging.info(f"SLURM job ID: {self.slurm_job_id}")
                return self.slurm_job_id
            else:
                logging.error("Failed to extract SLURM job ID.")
                return None
        except subprocess.CalledProcessError as e:
            logging.error(f"Error submitting SLURM job: {e}")
            return None

    def is_job_finished(self):
        """
        Check if the SLURM job has finished.
        """
        if self.slurm_job_id is None:
            logging.error("Error: SLURM job ID is not set.")
            return False

        try:
            username = subprocess.check_output("whoami", text=True).strip()
            command = f"squeue -u {username} -h -o %i"  # Check if job ID exists in queue
            output = subprocess.check_output(command, shell=True, text=True)
            job_ids = output.splitlines()

            if self.slurm_job_id in job_ids:
                return False  # Job is still running
            return True  # Job is finished
        except subprocess.CalledProcessError as e:
            logging.error(f"Error checking job status: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return False

    def monitor_slurm_job(self):
        """
        Monitor the SLURM job and construct the model path after completion.
        """
        if self.slurm_job_id is None:
            logging.error("Error: SLURM job ID is not set.")
            return None

        while not self.is_job_finished():
            logging.info(f"SLURM job {self.slurm_job_id} is still running. Checking again in 60 seconds...")
            time.sleep(30)  # Check every 30 seconds

        logging.info(f"SLURM job {self.slurm_job_id} has finished.")

        # Construct model path using fixed components from YAML config
        model_name = self.yaml_config.get("name", "")
        model_dir = self.yaml_config.get("model_dir", "")

        if not model_name or not model_dir:
            logging.error("Error: 'name' or 'model_dir' is not defined in the YAML config.")
            return None

        model_path = f"{model_dir}/{model_name}_run-123_stagetwo.model"

        if Path(model_path).exists():
            logging.info(f"Constructed model path: {model_path}")
            return model_path
        else:
            logging.error(f"Error: Model path does not exist: {model_path}")
            return None
