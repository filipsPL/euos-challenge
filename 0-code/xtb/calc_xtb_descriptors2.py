#!/usr/bin/env python3
"""
Calculate xTB quantum chemical descriptors for EUOS25 Challenge.

Calculates GFN2-xTB properties including HOMO/LUMO energies, gap, dipole moment,
and optionally sTDA excitation energies for absorption prediction.

Requirements:
    - xtb installed and in PATH (conda install -c conda-forge xtb)
    - rdkit for reading SDF files

Usage:
    # From SDF file with 3D conformers
    python calc_xtb_descriptors.py input.sdf output.csv [--stda]
    
    # From CSV with SMILES (generates 3D)
    python calc_xtb_descriptors.py input.csv output.csv [--stda]


Filip Stefaniak, fstefaniak@iimcb.gov.pl

"""

import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
import tempfile
import os
import re
from pathlib import Path
from tqdm import tqdm
import sys


def mol_to_xyz(mol, smiles=None):
    """
    Convert RDKit molecule with 3D coordinates to XYZ format.
    
    Args:
        mol: RDKit molecule object with 3D coordinates
        smiles: Optional SMILES string for comment line
    
    Returns:
        xyz_string: XYZ format string, or None if failed
    """
    try:
        if mol is None:
            return None
        
        # Check if molecule has 3D coordinates
        if mol.GetNumConformers() == 0:
            return None
        
        conf = mol.GetConformer()
        
        # Build XYZ string
        comment = smiles if smiles else "molecule"
        xyz_lines = [str(mol.GetNumAtoms()), comment]
        
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
        
        return "\n".join(xyz_lines)
    
    except Exception as e:
        print(f"Error converting molecule to XYZ: {e}", file=sys.stderr)
        return None


def smiles_to_xyz(smiles, optimize_3d=True):
    """
    Convert SMILES to XYZ format for xTB input.
    
    Args:
        smiles: SMILES string
        optimize_3d: Whether to do MMFF optimization before xTB
    
    Returns:
        xyz_string: XYZ format string, or None if failed
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            # Try again with different parameters
            if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42) != 0:
                return None
        
        # Optimize with MMFF if requested (faster than xTB optimization)
        if optimize_3d:
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except:
                pass  # Continue even if MMFF fails
        
        # Convert to XYZ format
        conf = mol.GetConformer()
        xyz_lines = [str(mol.GetNumAtoms()), smiles]
        
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
        
        return "\n".join(xyz_lines)
    
    except Exception as e:
        print(f"Error converting SMILES to XYZ: {e}", file=sys.stderr)
        return None


def parse_xtb_output(output_text):
    """
    Parse xTB output to extract relevant properties.
    
    Returns:
        dict with properties, or None if parsing failed
    """
    props = {}
    
    try:
        # HOMO-LUMO gap (eV)
        gap_match = re.search(r'HOMO-LUMO GAP\s+([-\d.]+)\s+eV', output_text)
        if gap_match:
            props['xtb_gap_eV'] = float(gap_match.group(1))
        
        # Alternative gap format in summary
        if 'xtb_gap_eV' not in props:
            gap_match2 = re.search(r'HL-Gap\s+([-\d.]+)\s+Eh\s+([-\d.]+)\s+eV', output_text)
            if gap_match2:
                props['xtb_gap_eV'] = float(gap_match2.group(2))
        
        # HOMO energy (eV)
        homo_match = re.search(r'\(HOMO\)\s+([-\d.]+)\s+eV', output_text)
        if homo_match:
            props['xtb_homo_eV'] = float(homo_match.group(1))
        
        # LUMO energy (eV)
        lumo_match = re.search(r'\(LUMO\)\s+([-\d.]+)\s+eV', output_text)
        if lumo_match:
            props['xtb_lumo_eV'] = float(lumo_match.group(1))
        
        # Fermi level (eV)
        fermi_match = re.search(r'Fermi-level\s+([-\d.]+)\s+Eh\s+([-\d.]+)\s+eV', output_text)
        if fermi_match:
            props['xtb_fermi_level_eV'] = float(fermi_match.group(2))
        
        # Total energy (Hartree)
        energy_match = re.search(r'TOTAL ENERGY\s+([-\d.]+)\s+Eh', output_text)
        if energy_match:
            props['xtb_total_energy_Eh'] = float(energy_match.group(1))
        
        # SCC energy
        scc_match = re.search(r'SCC energy\s+([-\d.]+)\s+Eh', output_text)
        if scc_match:
            props['xtb_scc_energy_Eh'] = float(scc_match.group(1))
        
        # Dispersion energy (important for aggregation/π-π stacking)
        disp_match = re.search(r'-> dispersion\s+([-\d.]+)\s+Eh', output_text)
        if disp_match:
            props['xtb_dispersion_Eh'] = float(disp_match.group(1))
        
        # Repulsion energy
        rep_match = re.search(r'repulsion energy\s+([-\d.]+)\s+Eh', output_text)
        if rep_match:
            props['xtb_repulsion_Eh'] = float(rep_match.group(1))
        
        # Electrostatic energy (isotropic)
        es_iso_match = re.search(r'-> isotropic ES\s+([-\d.]+)\s+Eh', output_text)
        if es_iso_match:
            props['xtb_isotropic_es_Eh'] = float(es_iso_match.group(1))
        
        # Electrostatic energy (anisotropic)
        es_aniso_match = re.search(r'-> anisotropic ES\s+([-\d.]+)\s+Eh', output_text)
        if es_aniso_match:
            props['xtb_anisotropic_es_Eh'] = float(es_aniso_match.group(1))
        
        # Dipole moment (Debye) - from molecular dipole section
        dipole_match = re.search(r'molecular dipole:.*?full:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', 
                                output_text, re.DOTALL)
        if dipole_match:
            props['xtb_dipole_D'] = float(dipole_match.group(4))
            props['xtb_dipole_x'] = float(dipole_match.group(1))
            props['xtb_dipole_y'] = float(dipole_match.group(2))
            props['xtb_dipole_z'] = float(dipole_match.group(3))
        
        # Molecular C6 coefficient (London dispersion, relates to π-system size)
        c6_match = re.search(r'Mol\. C6AA /au·bohr⁶\s*:\s*([-\d.]+)', output_text)
        if c6_match:
            props['xtb_mol_c6'] = float(c6_match.group(1))
        
        # Molecular C8 coefficient
        c8_match = re.search(r'Mol\. C8AA /au·bohr⁸\s*:\s*([-\d.]+)', output_text)
        if c8_match:
            props['xtb_mol_c8'] = float(c8_match.group(1))
        
        # Molecular polarizability (relates to absorption)
        alpha_match = re.search(r'Mol\. α\(0\) /au\s*:\s*([-\d.]+)', output_text)
        if alpha_match:
            props['xtb_polarizability'] = float(alpha_match.group(1))
        
        # Electronic temperature
        temp_match = re.search(r'electronic temp\.\s+([-\d.]+)\s+K', output_text)
        if temp_match:
            props['xtb_electronic_temp_K'] = float(temp_match.group(1))
        
        # Total charge
        charge_match = re.search(r'total charge\s+([-\d.]+)\s+e', output_text)
        if charge_match:
            props['xtb_total_charge'] = float(charge_match.group(1))
        
        # Gradient norm (measure of geometry quality/strain)
        grad_match = re.search(r'GRADIENT NORM\s+([-\d.]+)\s+Eh', output_text)
        if grad_match:
            props['xtb_gradient_norm'] = float(grad_match.group(1))
        
        # Molecular quadrupole moment (shape/charge distribution)
        # Extract the full quadrupole tensor diagonal elements
        quad_match = re.search(r'full:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
                              output_text.split('molecular quadrupole')[1] if 'molecular quadrupole' in output_text else '')
        if quad_match:
            props['xtb_quadrupole_xx'] = float(quad_match.group(1))
            props['xtb_quadrupole_xy'] = float(quad_match.group(2))
            props['xtb_quadrupole_yy'] = float(quad_match.group(3))
            props['xtb_quadrupole_xz'] = float(quad_match.group(4))
            props['xtb_quadrupole_yz'] = float(quad_match.group(5))
            props['xtb_quadrupole_zz'] = float(quad_match.group(6))
        
        # Derived properties
        if 'xtb_gap_eV' in props and props['xtb_gap_eV'] > 0:
            # Convert gap to approximate wavelength (nm)
            # E(eV) = 1240 / λ(nm)
            props['xtb_gap_to_wavelength_nm'] = 1240.0 / props['xtb_gap_eV']
        
        # HOMO-LUMO characteristics
        if 'xtb_homo_eV' in props and 'xtb_lumo_eV' in props:
            # Electronegativity (approximation)
            props['xtb_electronegativity_eV'] = -(props['xtb_homo_eV'] + props['xtb_lumo_eV']) / 2.0
            # Chemical hardness
            props['xtb_hardness_eV'] = (props['xtb_lumo_eV'] - props['xtb_homo_eV']) / 2.0
        
        return props if props else None
    
    except Exception as e:
        print(f"Error parsing xTB output: {e}", file=sys.stderr)
        return None


def parse_stda_output(output_text):
    """
    Parse sTDA-xTB output for excitation energies and oscillator strengths.
    
    Returns:
        dict with excitation properties
    """
    props = {}
    
    try:
        # Find the excitation energy table
        lines = output_text.split('\n')
        
        excitations = []
        oscillators = []
        wavelengths = []
        
        in_table = False
        for i, line in enumerate(lines):
            # Look for the sTDA results header - more flexible pattern
            if 'excitation energies' in line.lower() or \
               'sTDA-xTB' in line or \
               'state    eV      nm       fL' in line.lower():
                in_table = True
                continue
            
            # Alternative: look for typical sTDA table start
            if 'state' in line.lower() and 'ev' in line.lower() and 'nm' in line.lower():
                in_table = True
                continue
            
            if in_table:
                # Parse lines like: "  1   3.456   358.7  0.1234  ..."
                # or "    1    3.456    358.7   0.1234"
                match = re.match(r'\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
                if match:
                    state = int(match.group(1))
                    energy_eV = float(match.group(2))
                    wavelength_nm = float(match.group(3))
                    osc_strength = float(match.group(4))
                    
                    excitations.append(energy_eV)
                    wavelengths.append(wavelength_nm)
                    oscillators.append(osc_strength)
                    
                    if state >= 20:  # Get first 20 transitions
                        break
                
                # Stop at empty line after we've found some transitions
                elif line.strip() == '' and excitations:
                    break
                
                # Stop if we hit next section
                elif 'eigenvalue' in line.lower() or 'covergence' in line.lower():
                    break
        
        if excitations:
            # First excitation (S0 -> S1, usually most important)
            props['stda_S1_energy_eV'] = excitations[0]
            props['stda_S1_wavelength_nm'] = wavelengths[0]
            props['stda_S1_osc_strength'] = oscillators[0]
            
            # Strongest absorption (max oscillator strength)
            max_idx = np.argmax(oscillators)
            props['stda_max_osc_energy_eV'] = excitations[max_idx]
            props['stda_max_osc_wavelength_nm'] = wavelengths[max_idx]
            props['stda_max_osc_strength'] = oscillators[max_idx]
            
            # Sum of oscillator strengths (total absorption intensity)
            props['stda_sum_osc_strength'] = sum(oscillators)
            
            # Number of strong transitions (osc > 0.01)
            props['stda_num_strong_transitions'] = sum(1 for o in oscillators if o > 0.01)
            
            # Count transitions in UV-Vis range (200-800 nm)
            props['stda_num_uvvis_transitions'] = sum(1 for w in wavelengths if 200 <= w <= 800)
            
            # Longest wavelength transition (lowest energy)
            props['stda_longest_wavelength_nm'] = max(wavelengths)
            props['stda_lowest_energy_eV'] = min(excitations)
        
        return props if props else None
    
    except Exception as e:
        print(f"Error parsing sTDA output: {e}", file=sys.stderr)
        return None


def run_xtb(xyz_string, use_stda=False, timeout=120, debug=False, optimize='crude'):
    """
    Run xTB calculation on molecule.
    
    Args:
        xyz_string: XYZ format molecule
        use_stda: Whether to calculate excited states with sTDA (DEPRECATED in xTB 6.7+)
        timeout: Maximum time in seconds
        debug: Print detailed output for debugging
        optimize: Optimization level - 'none', 'crude', 'normal', 'tight'
    
    Returns:
        dict with properties, or None if failed
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write XYZ file
            xyz_file = os.path.join(tmpdir, 'mol.xyz')
            with open(xyz_file, 'w') as f:
                f.write(xyz_string)
            
            # Build xTB command
            cmd = ['xtb', 'mol.xyz', '--gfn', '2', '--parallel', '1']
            
            # Add optimization if requested
            if optimize and optimize != 'none':
                cmd.extend(['--opt', optimize])
                if debug:
                    print(f"\nRunning xTB with {optimize} optimization: {' '.join(cmd)}")
            else:
                if debug:
                    print(f"\nRunning xTB (no optimization): {' '.join(cmd)}")
            
            # Run xTB
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                if debug:
                    print(f"xTB failed with return code {result.returncode}")
                    print(f"STDERR: {result.stderr[:500]}")
                return None
            
            # Parse output
            props = parse_xtb_output(result.stdout)
            
            if debug:
                print(f"Extracted {len(props) if props else 0} properties")
                if props:
                    print(f"Properties: {list(props.keys())[:10]}...")  # Show first 10
                    if 'xtb_gradient_norm' in props:
                        print(f"Final gradient norm: {props['xtb_gradient_norm']:.6f} Eh/Bohr")
            
            # Note about sTDA
            if use_stda and debug:
                print("\nNOTE: sTDA flag was provided but is not supported in xTB 6.7+")
                print("      To calculate excited states, you need the separate 'stda' program")
                print("      See: https://github.com/grimme-lab/stda")
            
            return props
    
    except subprocess.TimeoutExpired:
        print("xTB calculation timeout", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error running xTB: {e}", file=sys.stderr)
        if debug:
            import traceback
            traceback.print_exc()
        return None
    """
    Run xTB calculation on molecule.
    
    Args:
        xyz_string: XYZ format molecule
        use_stda: Whether to calculate excited states with sTDA
        timeout: Maximum time in seconds
        debug: Print detailed output for debugging
    
    Returns:
        dict with properties, or None if failed
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write XYZ file
            xyz_file = os.path.join(tmpdir, 'mol.xyz')
            with open(xyz_file, 'w') as f:
                f.write(xyz_string)
            
            # Build xTB command
            cmd = ['xtb', 'mol.xyz', '--gfn', '2', '--parallel', '1']
            
            if debug:
                print(f"\nRunning ground state xTB: {' '.join(cmd)}")
            
            # Run xTB
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                if debug:
                    print(f"xTB failed with return code {result.returncode}")
                    print(f"STDERR: {result.stderr[:500]}")
                return None
            
            # Parse output
            props = parse_xtb_output(result.stdout)
            
            if debug:
                print(f"Ground state properties: {props}")
            
            # Run sTDA if requested (separate step!)
            if use_stda and props:
                # sTDA in xTB 6.7+ requires running optimization first, then sTDA
                # The --stda flag needs the wavefunction files from previous run
                
                # First, check if xtbrestart exists (created by first run)
                xtbrestart_file = os.path.join(tmpdir, 'xtbrestart')
                
                if not os.path.exists(xtbrestart_file):
                    # Run a quick optimization to generate necessary files
                    opt_cmd = ['xtb', 'mol.xyz', '--gfn', '2', '--opt', 'crude']
                    if debug:
                        print(f"\nRunning optimization for sTDA: {' '.join(opt_cmd)}")
                    
                    opt_result = subprocess.run(
                        opt_cmd,
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    
                    if debug:
                        print(f"Optimization return code: {opt_result.returncode}")
                
                # Now run sTDA
                # Try different sTDA command formats
                stda_commands = [
                    ['xtb', 'mol.xyz', '--gfn', '2', '--stda', '--norestart'],
                    ['stda_xtb', 'mol.xyz'],  # Older xTB versions
                    ['xtb', 'mol.xyz', '--gfn', '2', '--stda']
                ]
                
                stda_success = False
                for stda_cmd in stda_commands:
                    if debug:
                        print(f"\nTrying sTDA command: {' '.join(stda_cmd)}")
                    
                    try:
                        stda_result = subprocess.run(
                            stda_cmd,
                            cwd=tmpdir,
                            capture_output=True,
                            text=True,
                            timeout=timeout
                        )
                        
                        if debug:
                            print(f"sTDA return code: {stda_result.returncode}")
                        
                        if stda_result.returncode == 0:
                            # Check if output contains excitation data
                            if 'excitation' in stda_result.stdout.lower() or \
                               'sTDA' in stda_result.stdout:
                                
                                if debug:
                                    # Save full output to file for inspection
                                    debug_file = 'xtb_stda_debug_output.txt'
                                    with open(debug_file, 'w') as f:
                                        f.write("=== COMMAND ===\n")
                                        f.write(' '.join(stda_cmd) + '\n\n')
                                        f.write("=== STDOUT ===\n")
                                        f.write(stda_result.stdout)
                                        f.write("\n\n=== STDERR ===\n")
                                        f.write(stda_result.stderr)
                                    print(f"Full sTDA output saved to: {debug_file}")
                                    print(f"sTDA output length: {len(stda_result.stdout)} characters")
                                    
                                    # Look for key phrases in output
                                    if 'sTDA-xTB' in stda_result.stdout:
                                        print("  ✓ Found 'sTDA-xTB' in output")
                                    if 'excitation' in stda_result.stdout.lower():
                                        print("  ✓ Found 'excitation' in output")
                                    
                                    # Show context around 'excitation' if found
                                    exc_pos = stda_result.stdout.lower().find('excitation')
                                    if exc_pos > 0:
                                        start = max(0, exc_pos - 200)
                                        end = min(len(stda_result.stdout), exc_pos + 1000)
                                        print(f"\nContext around 'excitation' keyword:\n")
                                        print(stda_result.stdout[start:end])
                                        print("\n" + "="*60)
                                
                                stda_props = parse_stda_output(stda_result.stdout)
                                if debug:
                                    print(f"Parsed sTDA properties: {stda_props}")
                                
                                if stda_props:
                                    props.update(stda_props)
                                    stda_success = True
                                    break
                                elif debug:
                                    print("WARNING: sTDA output found but no properties were parsed")
                    
                    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                        if debug:
                            print(f"sTDA command failed: {e}")
                        continue
                
                if not stda_success and debug:
                    print("\nWARNING: All sTDA command variants failed or produced no parseable output")
                    print("This may indicate:")
                    print("  1. sTDA is not available in your xTB version")
                    print("  2. The molecule is too large for sTDA")
                    print("  3. sTDA requires additional setup")
                    print("\nTo check: run manually 'xtb mol.xyz --gfn 2 --stda' and inspect output")
            
            return props
    
    except subprocess.TimeoutExpired:
        print("xTB calculation timeout", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error running xTB: {e}", file=sys.stderr)
        if debug:
            import traceback
            traceback.print_exc()
        return None


def read_sdf_file(sdf_file, max_molecules=None):
    """
    Read molecules from SDF file.
    
    Args:
        sdf_file: Path to SDF file
        max_molecules: Maximum number of molecules to read (None = all)
    
    Returns:
        List of (mol, smiles) tuples
    """
    molecules = []
    
    print(f"Reading SDF file: {sdf_file}")
    
    # Handle both compressed and uncompressed SDF files
    if str(sdf_file).endswith('.gz'):
        import gzip
        with gzip.open(sdf_file, 'rb') as f:
            suppl = Chem.ForwardSDMolSupplier(f, removeHs=False)
            for i, mol in enumerate(suppl):
                if max_molecules and i >= max_molecules:
                    break
                    
                if mol is None:
                    print(f"Warning: Could not read molecule {i}", file=sys.stderr)
                    molecules.append((None, None))
                    continue
                
                # Try to get SMILES from properties or generate it
                smiles = None
                if mol.HasProp('SMILES'):
                    smiles = mol.GetProp('SMILES')
                elif mol.HasProp('smiles'):
                    smiles = mol.GetProp('smiles')
                else:
                    # Generate SMILES from molecule
                    try:
                        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
                    except:
                        smiles = f"mol_{i}"
                
                molecules.append((mol, smiles))
    else:
        suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
        
        for i, mol in enumerate(suppl):
            if max_molecules and i >= max_molecules:
                break
                
            if mol is None:
                print(f"Warning: Could not read molecule {i}", file=sys.stderr)
                molecules.append((None, None))
                continue
            
            # Try to get SMILES from properties or generate it
            smiles = None
            if mol.HasProp('SMILES'):
                smiles = mol.GetProp('SMILES')
            elif mol.HasProp('smiles'):
                smiles = mol.GetProp('smiles')
            else:
                # Generate SMILES from molecule
                try:
                    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
                except:
                    smiles = f"mol_{i}"
            
            molecules.append((mol, smiles))
    
    print(f"Read {len(molecules)} molecules from SDF")
    return molecules


def calculate_xtb_descriptors_from_mols(molecules, use_stda=False, debug=False, optimize='crude'):
    """
    Calculate xTB descriptors from RDKit molecule objects with 3D coordinates.
    
    Args:
        molecules: List of (mol, smiles) tuples
        use_stda: Calculate excited states (slower)
        debug: Enable debug output for first molecule
        optimize: Optimization level - 'none', 'crude', 'normal', 'tight'
    
    Returns:
        List of descriptor dictionaries
    """
    results = []
    
    print(f"Calculating xTB descriptors for {len(molecules)} molecules...")
    print(f"Optimization: {optimize}")
    print(f"sTDA excited states: {'ENABLED' if use_stda else 'DISABLED'}")
    
    stda_success_count = 0
    
    for idx, (mol, smiles) in enumerate(tqdm(molecules, desc="xTB calculations")):
        if mol is None:
            results.append({})
            continue
        
        # Convert to XYZ
        xyz = mol_to_xyz(mol, smiles)
        
        if xyz is None:
            results.append({})
            continue
        
        # Run xTB (enable debug for first molecule only)
        props = run_xtb(xyz, use_stda=use_stda, debug=(debug and idx == 0), optimize=optimize)
        
        if props is None:
            results.append({})
        else:
            # Check if sTDA properties were calculated
            if use_stda and any(key.startswith('stda_') for key in props.keys()):
                stda_success_count += 1
            results.append(props)
    
    if use_stda:
        print(f"\nsTDA calculations successful: {stda_success_count}/{len(molecules)}")
    
    return results


def calculate_xtb_descriptors(smiles_list, use_stda=False, optimize_3d=True, optimize='crude'):
    """
    Calculate xTB descriptors for a list of SMILES (generates 3D on the fly).
    
    Args:
        smiles_list: List of SMILES strings
        use_stda: Calculate excited states (slower)
        optimize_3d: Do MMFF optimization before xTB
        optimize: xTB optimization level - 'none', 'crude', 'normal', 'tight'
    
    Returns:
        List of descriptor dictionaries
    """
    results = []
    
    print(f"Calculating xTB descriptors for {len(smiles_list)} molecules...")
    print(f"MMFF pre-optimization: {optimize_3d}")
    print(f"xTB optimization: {optimize}")
    print(f"sTDA excited states: {'ENABLED' if use_stda else 'DISABLED'}")
    
    stda_success_count = 0
    
    for smiles in tqdm(smiles_list, desc="xTB calculations"):
        # Convert to XYZ
        xyz = smiles_to_xyz(smiles, optimize_3d=optimize_3d)
        
        if xyz is None:
            results.append({})
            continue
        
        # Run xTB
        props = run_xtb(xyz, use_stda=use_stda, optimize=optimize)
        
        if props is None:
            results.append({})
        else:
            # Check if sTDA properties were calculated
            if use_stda and any(key.startswith('stda_') for key in props.keys()):
                stda_success_count += 1
            results.append(props)
    
    if use_stda:
        print(f"\nsTDA calculations successful: {stda_success_count}/{len(smiles_list)}")
    
    return results


def check_xtb_installed():
    """Check if xTB is installed and accessible."""
    try:
        result = subprocess.run(
            ['xtb', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"Found xTB: {version}")
            return True
        return False
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Calculate xTB quantum chemical descriptors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From SDF file with pre-calculated 3D conformers (default: crude optimization)
    python calc_xtb_descriptors.py molecules.sdf output.csv
    
    # Without optimization (faster)
    python calc_xtb_descriptors.py molecules.sdf output.csv --optimize none
    
    # With tighter optimization (slower but more accurate)
    python calc_xtb_descriptors.py molecules.sdf output.csv --optimize normal
    
    # Debug first molecule
    python calc_xtb_descriptors.py molecules.sdf output.csv --debug
    
    # From CSV with SMILES (generates 3D on the fly)
    python calc_xtb_descriptors.py input.csv output.csv --smiles-col SMILES
        """
    )
    parser.add_argument('input_file', help='Input SDF or CSV file')
    parser.add_argument('output_csv', help='Output CSV with xTB descriptors')
    parser.add_argument('--smiles-col', default='SMILES', 
                        help='Name of SMILES column (for CSV input, default: SMILES)')
    parser.add_argument('--stda', action='store_true',
                        help='Calculate excited states with sTDA (DEPRECATED in xTB 6.7+)')
    parser.add_argument('--optimize', choices=['none', 'crude', 'normal', 'tight'], 
                        default='crude',
                        help='xTB geometry optimization level (default: crude)')
    parser.add_argument('--no-optimize', action='store_true',
                        help='DEPRECATED: Use --optimize none instead')
    parser.add_argument('--max-molecules', type=int, default=None,
                        help='Limit number of molecules (for testing)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for first molecule to diagnose issues')
    
    args = parser.parse_args()
    
    # Handle deprecated --no-optimize flag
    if args.no_optimize:
        args.optimize = 'none'
        print("WARNING: --no-optimize is deprecated, use --optimize none instead")
    
    # Check if xTB is installed
    print("Checking xTB installation...")
    if not check_xtb_installed():
        print("\nERROR: xTB not found in PATH!")
        print("\nInstall xTB with:")
        print("  conda install -c conda-forge xtb")
        print("\nOr visit: https://github.com/grimme-lab/xtb")
        return 1
    
    # Determine input file type
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input_file}")
        return 1
    
    input_name = input_path.name.lower()
    # Check for SDF extension (including .sdf.gz)
    is_sdf = input_name.endswith('.sdf') or input_name.endswith('.sd') or \
             input_name.endswith('.sdf.gz') or input_name.endswith('.sd.gz')
    
    # Process based on input type
    if is_sdf:
        print(f"\nDetected SDF input: {args.input_file}")
        print("Using pre-calculated 3D conformers from SDF")
        
        # Read SDF file
        molecules = read_sdf_file(args.input_file, max_molecules=args.max_molecules)
        
        if not molecules:
            print("ERROR: No molecules read from SDF file")
            return 1
        
        # Calculate descriptors
        print("\nCalculating xTB descriptors...")
        if args.optimize != 'none':
            print(f"With {args.optimize} geometry optimization")
            print("This may take a while (estimate: ~5-15 seconds per molecule)")
            print(f"Total estimated time: {len(molecules) * 10 / 60:.1f} minutes")
        else:
            print("Without geometry optimization (faster)")
            print("This may take a while (estimate: ~1-3 seconds per molecule)")
            print(f"Total estimated time: {len(molecules) * 2 / 60:.1f} minutes")
        
        descriptors = calculate_xtb_descriptors_from_mols(
            molecules,
            use_stda=args.stda,
            debug=args.debug,
            optimize=args.optimize
        )
        
        # Create output dataframe
        df_desc = pd.DataFrame(descriptors)
        
        # Extract SMILES from molecules
        smiles_list = [smiles if smiles else f"mol_{i}" for i, (mol, smiles) in enumerate(molecules)]
        df_output = pd.DataFrame({'SMILES': smiles_list})
        df_output = pd.concat([df_output, df_desc], axis=1)
        
        # Report statistics
        print("\nDescriptor calculation complete!")
        print(f"Successfully calculated: {df_desc.notna().any(axis=1).sum()}/{len(df_desc)} molecules")
        
    else:
        # CSV input path
        print(f"\nDetected CSV input: {args.input_file}")
        print("Will generate 3D conformers from SMILES")
        
        # Load input CSV
        try:
            df = pd.read_csv(args.input_file, compression='gzip' if args.input_file.endswith('.gz') else None)
            print(f"Loaded {len(df)} molecules")
        except Exception as e:
            print(f"ERROR loading input: {e}")
            return 1
        
        # Check SMILES column
        if args.smiles_col not in df.columns:
            print(f"ERROR: Column '{args.smiles_col}' not found in input CSV")
            print(f"Available columns: {list(df.columns)}")
            return 1
        
        # Limit molecules if requested
        if args.max_molecules:
            print(f"Limiting to first {args.max_molecules} molecules")
            df = df.head(args.max_molecules)
        
        # Calculate descriptors
        print("\nCalculating xTB descriptors...")
        if args.optimize != 'none':
            print(f"With {args.optimize} geometry optimization")
            print("This may take a while (estimate: ~5-15 seconds per molecule)")
            print(f"Total estimated time: {len(df) * 10 / 60:.1f} minutes")
        else:
            print("Without geometry optimization (faster)")
            print("This may take a while (estimate: ~1-3 seconds per molecule)")
            print(f"Total estimated time: {len(df) * 2 / 60:.1f} minutes")
        
        descriptors = calculate_xtb_descriptors(
            df[args.smiles_col].tolist(),
            use_stda=args.stda,
            optimize_3d=not args.no_optimize,
            optimize=args.optimize
        )
        
        # Create output dataframe
        df_desc = pd.DataFrame(descriptors)
        
        # Combine with SMILES
        df_output = pd.concat([df[[args.smiles_col]], df_desc], axis=1)
        
        # Report statistics
        print("\nDescriptor calculation complete!")
        print(f"Successfully calculated: {df_desc.notna().any(axis=1).sum()}/{len(df)} molecules")
    
    # Common reporting for both paths
    if not df_desc.empty:
        print(f"Calculated descriptors: {list(df_desc.columns)}")
    else:
        print("WARNING: No descriptors were calculated!")
    
    # Save output
    print(f"\nSaving to: {args.output_csv}")
    compression = 'gzip' if args.output_csv.endswith('.gz') else None
    df_output.to_csv(args.output_csv, index=False, compression=compression, float_format='%.6f')
    
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())


