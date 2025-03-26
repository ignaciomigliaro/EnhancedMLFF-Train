import os
import logging
from ase.io import write

class DFTInputGenerator:
    """Generates DFT input files for ORCA and Quantum ESPRESSO (QE)."""

    def __init__(self, output_dir, dft_software, template_dir="templates"):
        """
        Initializes the DFTInputGenerator.

        Parameters:
        - output_dir (str): Directory where input files will be saved.
        - dft_software (str): Specifies whether to generate ORCA or QE inputs.
        - template_dir (str): Directory containing SLURM templates.
        """
        self.output_dir = output_dir
        self.dft_software = dft_software.lower()
        self.template_dir = template_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_dft_inputs(self, atoms_list):
        """
        Generates DFT input files and corresponding SLURM scripts.

        Parameters:
        - atoms_list (list): List of ASE Atoms objects.

        Returns:
        - input_files (list): List of generated input file paths.
        """
        if self.dft_software == "orca":
            template = os.path.join(self.template_dir, "orca_template.inp")
            inputs = self._write_orca_files(atoms_list, template)
            self.create_slurm_scripts(inputs, "slurm_template_orca.slurm")
        elif self.dft_software == "qe":
            inputs = self._write_qe_files(atoms_list)
            self.create_slurm_scripts(inputs, "slurm_template_qe.slurm")
        else:
            raise ValueError(f"Unsupported DFT software: {self.dft_software}")

        return inputs

    def _write_orca_files(self, atoms_list, template):
        """
        Generates ORCA input files.

        Parameters:
            atoms_list (list): List of ASE Atoms objects.
            template (str): Path to the ORCA template file.

        Returns:
            input_files (list): List of generated ORCA input file paths.
        """
        input_files = []

        logging.info("Writing ORCA input files.")

        for i, atoms in enumerate(atoms_list):
            base_name = f"structure_{i}"
            xyz_file = os.path.join(self.output_dir, f"{base_name}.xyz")
            input_file = os.path.join(self.output_dir, f"{base_name}.inp")

            # Write XYZ file
            write(xyz_file, atoms, format="xyz")

            # Read the ORCA template
            with open(template, "r") as tmpl:
                content = tmpl.read()

            # Append the XYZ file reference
            formatted_content = content + f"\n\n* xyzfile 0 1 {base_name}.xyz\n\n"

            # Write ORCA input file
            with open(input_file, "w") as inp:
                inp.write(formatted_content)

            input_files.append(input_file)
            logging.info(f"Created ORCA input: {input_file}, referencing {xyz_file}")

        return input_files

    def _write_qe_files(self, atoms_list):
        """
        Generates Quantum ESPRESSO input files.

        Parameters:
            atoms_list (list): List of ASE Atoms objects.

        Returns:
            input_files (list): List of generated QE input file paths.
        """
        input_data = {
            "calculation": "scf",
            "prefix": "qe_input",
            "pseudo_dir": "~/QE/pseudo",
            "outdir": "./out/",
            "verbosity": "high",
            "tstress": True,
            "tprnfor": True,
            "ecutrho": 600,
            "ecutwfc": 90,
            "vdw_corr": "mbd",
            "occupations": "smearing",
            "smearing": "cold",
        }
        
        pseudos = {
            "Cl": "Cl.upf",
            "O": "O.upf",
            "F": "F.upf",
            "I": "I.upf",
            "Br": "Br.upf",
            "La": "La.upf",
            "Li": "Li.upf",
            "Zr": "Zr.upf",
            "C": "C.upf",
            "H": "H.upf",
            "Nb": "Nb.upf",
        }

        input_files = []

        logging.info("Writing QE input files.")

        for i, atoms in enumerate(atoms_list):
            filename = os.path.join(self.output_dir, f"qe_input_{i}.in")
            
            write(
                filename=filename,
                images=atoms, 
                format='espresso-in',
                input_data=input_data,
                pseudopotentials=pseudos,
                kspacing=0.05
            )

            input_files.append(filename)
            logging.info(f"Created QE input: {filename}")

        return input_files
    
    def create_slurm_scripts(self, input_files, template_name):
        """
        Generate SLURM job scripts for multiple input files.

        Args:
            input_files (list): List of DFT input file paths.
            template_name (str): Name of the SLURM template file (e.g., "slurm_template_orca.slurm").
        """
        for input_file in input_files:
            base_name = os.path.basename(input_file)  # Extract filename (with extension)
            filename_no_ext, ext = os.path.splitext(base_name)  # Separate name and extension
            slurm_name = filename_no_ext + ".slurm"  # Proper SLURM script name
            self.create_slurm_script(template_name, slurm_name, base_name, filename_no_ext)

    def create_slurm_script(self, template_name, output_name, input_filename, output_filename):
        """
        Generate a single SLURM job script from a template.

        Args:
            template_name (str): Name of the SLURM template file (e.g., "slurm_template_orca.slurm").
            output_name (str): Name of the output SLURM script.
            input_filename (str): Full input filename (e.g., "structure_0.inp").
            output_filename (str): Filename without extension (e.g., "structure_0").
        """
        template_path = os.path.join(self.template_dir, template_name)
        output_path = os.path.join(self.output_dir, output_name)

        # Ensure the template exists
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file '{template_path}' not found.")

        # Read the SLURM template
        with open(template_path, "r") as file:
            slurm_content = file.read()

        # Replace placeholders
        slurm_content = slurm_content.replace("{filename}", input_filename)  # Keep extension for input
        slurm_content = slurm_content.replace("{output_filename}", output_filename)  # Remove extension for output

        # Write the modified content to the output SLURM script
        with open(output_path, "w") as file:
            file.write(slurm_content)

        logging.info(f"Created SLURM script: {output_path}")