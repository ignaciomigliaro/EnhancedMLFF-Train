import os
import logging
from ase.io import write

class DFTInputGenerator:
    """Generates DFT input files for ORCA and Quantum ESPRESSO (QE)."""

    def __init__(self, output_dir):
        """
        Initializes the DFTInputGenerator.

        Parameters:
        - output_dir (str): Directory where input files will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_orca_files(self, atoms_list, template):
        """
        Generates ORCA input files from a list of ASE Atoms objects.

        Parameters:
            atoms_list (list): List of ASE Atoms objects.
            template (str): Path to the ORCA template file.

        Returns:
            input_files (list): List of generated ORCA input file paths.
            xyz_files (list): List of generated XYZ file paths.
        """
        input_files = []
        xyz_files = []

        logging.info("Writing XYZ files and ORCA input files.")

        for i, atoms in enumerate(atoms_list):
            base_name = f"structure_{i}"
            xyz_file = os.path.join(self.output_dir, f"{base_name}.xyz")
            input_file = os.path.join(self.output_dir, f"{base_name}.inp")

            # Write XYZ file
            write(xyz_file, atoms, format="xyz")
            xyz_files.append(xyz_file)

            # Read the ORCA template
            with open(template, "r") as tmpl:
                content = tmpl.read()

            # Append the XYZ file reference
            formatted_content = content + f"\n\n* xyzfile 0 1 {base_name}.xyz\n\n"

            # Write ORCA input file
            with open(input_file, "w") as inp:
                inp.write(formatted_content)

            input_files.append(input_file)
            logging.info(f"Created {input_file} referencing {xyz_file}")

        return input_files, xyz_files

    def write_qe_files(self, atoms_list):
        """
        Generates Quantum ESPRESSO input files from a list of ASE Atoms objects.

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
            "etot_conv_thr": 1.0e-03,
            "tstress": True,
            "tprnfor": True,
            "degauss": 1.4699723600e-02,
            "ecutrho": 600,
            "ecutwfc": 90,
            "vdw_corr": "mbd",
            "occupations": "smearing",
            "smearing": "cold",
            "electron_maxstep": 80,
            "mixing_beta": 4.0e-01,
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
            logging.info(f"Written {filename}")

        return input_files
