import importlib
from importlib import resources

import ase
from ase import Atoms
import numpy as np
import pytest
from pytest import approx

from functionalise_mof import find_pattern_in_structure, replace_pattern_in_structure, translate_molecule_origin, translate_replace_pattern, rotate_replace_pattern, replace_pattern_orient
import tests

from scipy.spatial.transform import Rotation as R

@pytest.fixture
def octane():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as path:
        structure = ase.io.read(path)
        structure.positions += 30
        structure.set_cell(60 * np.identity(3))
        yield structure

@pytest.fixture
def hkust1_cif():
    with importlib.resources.path(tests, "HKUST-1_withbonds.cif") as path:
        yield ase.io.read(path)

@pytest.fixture
def hkust1_3x3x3_xyz():
    with importlib.resources.path(tests, "HKUST-1_3x3x3.xyz") as path:
        structure = ase.io.read(path)
        structure.set_cell(79.0290 * np.identity(3))
        yield structure

@pytest.fixture
def hkust1_3x3x3_cif():
    with importlib.resources.path(tests, "HKUST-1_3x3x3.cif") as path:
        yield ase.io.read(path)

@pytest.fixture
def benzene():
    with importlib.resources.path(tests, "benzene.xyz") as path:
        yield ase.io.read(path)


def test_find_pattern_in_structure__octane_has_8_carbons(octane):
    pattern = Atoms('C', positions=[(0, 0, 0)])
    match_indices, match_atoms = find_pattern_in_structure(octane, pattern)
    assert len(match_indices) == 8
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ["C"]

def test_find_pattern_in_structure__octane_has_2_CH3(octane):
    pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    match_indices, match_atoms = find_pattern_in_structure(octane, pattern)
    assert len(match_atoms) == 2
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ["C", "H", "H", "H"]
        cpos = pattern_found[0].position
        assert ((pattern_found[1].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found[2].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found[3].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)

def test_find_pattern_in_structure__octane_has_12_CH2(octane):
    # there are technically 12 matches, since each CH3 makes 3 variations of CH2
    pattern = Atoms('CHH', positions=[(0, 0, 0),(-0.1  , -0.379, -1.017), (-0.547, -0.647,  0.685)])
    match_indices, match_atoms = find_pattern_in_structure(octane, pattern)

    assert len(match_atoms) == 12
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ["C", "H", "H"]
        cpos = pattern_found[0].position
        assert ((pattern_found[1].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)
        assert ((pattern_found[2].position - cpos) ** 2).sum() == approx(1.18704299, 5e-2)

def test_find_pattern_in_structure__octane_over_pbc_has_2_CH3():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)#[0:4]
        # move atoms across corner boundary
        structure.positions += -1.8
        # move coordinates into main 15 Å unit cell
        structure.positions %= 15
        structure.set_cell(15 * np.identity(3))

    pattern = Atoms('CHHH', positions=[(0, 0, 0), (-0.538, -0.635,  0.672), (-0.397,  0.993,  0.052), (-0.099, -0.371, -0.998)])
    match_indices, match_atoms = find_pattern_in_structure(structure, pattern)
    assert len(match_atoms) == 2
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ["C", "H", "H", "H"]

def test_find_pattern_in_structure__hkust1_unit_cell_has_32_benzene_rings(hkust1_cif, benzene):
    match_indices, match_atoms = find_pattern_in_structure(hkust1_cif, benzene)

    assert len(match_atoms) == 32
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['C','C','C','C','C','C','H','H','H']
        assert ((pattern_found[0].position - pattern_found[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
        assert ((pattern_found[0].position - pattern_found[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
        assert ((pattern_found[0].position - pattern_found[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
        assert ((pattern_found[5].position - pattern_found[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_find_pattern_in_structure__hkust1_unit_cell_has_48_Cu_metal_nodes(hkust1_cif):
    pattern = Atoms('Cu', positions=[(0, 0, 0)])
    match_indices, match_atoms = find_pattern_in_structure(hkust1_cif, pattern)

    assert len(match_atoms) == 48
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['Cu']

@pytest.mark.slow
def test_find_pattern_in_structure__hkust1_cif_3x3x3_supercell_has_864_benzene_rings(hkust1_3x3x3_cif, benzene):
    match_indices, match_atoms = find_pattern_in_structure(hkust1_3x3x3_cif, benzene)

    assert len(match_atoms) == 864
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['C','C','C','C','C','C','H','H','H']
        assert ((pattern_found[0].position - pattern_found[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
        assert ((pattern_found[0].position - pattern_found[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
        assert ((pattern_found[0].position - pattern_found[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
        assert ((pattern_found[5].position - pattern_found[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)


@pytest.mark.slow
def test_find_pattern_in_structure__hkust1_xyz_3x3x3_supercell_has_864_benzene_rings(hkust1_3x3x3_xyz, benzene):
    match_indices, match_atoms = find_pattern_in_structure(hkust1_3x3x3_xyz, benzene)

    assert len(match_atoms) == 864
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['C','C','C','C','C','C','H','H','H']
        assert ((pattern_found[0].position - pattern_found[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
        assert ((pattern_found[0].position - pattern_found[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
        assert ((pattern_found[0].position - pattern_found[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
        assert ((pattern_found[5].position - pattern_found[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

@pytest.mark.slow
def test_find_pattern_in_structure__hkust1_cif_3x3x3_supercell_has_1296_Cu_metal_nodes(hkust1_3x3x3_cif):
    pattern = Atoms('Cu', positions=[(0, 0, 0)])
    match_indices, match_atoms = find_pattern_in_structure(hkust1_3x3x3_cif, pattern)

    assert len(match_atoms) == 1296
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['Cu']

def test_find_pattern_in_structure__hkust1_xyz_3x3x3_supercell_has_1296_Cu_metal_nodes(hkust1_3x3x3_xyz):
    pattern = Atoms('Cu', positions=[(0, 0, 0)])
    match_indices, match_atoms = find_pattern_in_structure(hkust1_3x3x3_xyz, pattern)

    assert len(match_atoms) == 1296
    for pattern_found in match_atoms:
        assert pattern_found.get_chemical_symbols() == ['Cu']

@pytest.mark.skip(reason="replace not implemented yet")
def test_replace_pattern_in_structure__replace_hydrogens_in_octane_with_nothing():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)
    search_pattern = Atoms('H', positions=[(0, 0, 0)])
    replace_pattern = search_pattern

    replaced_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)
    assert len(replaced_structure) == 8
    assert replaced_structure.get_chemical_symbols() == ["C"] * 8

@pytest.mark.skip(reason="replace not implemented yet")
def test_replace_pattern_in_structure__replace_hydrogens_in_octane_with_hydrogens():
    # CH3 CH2 CH2 CH2 CH2 CH2 CH2 CH3 #
    with importlib.resources.path(tests, "octane.xyz") as octane_path:
        structure = ase.io.read(octane_path)
    search_pattern = Atoms('H', positions=[(0, 0, 0)])
    replace_pattern = search_pattern

    replaced_structure = replace_pattern_in_structure(structure, search_pattern, replace_pattern)
    assert len(replaced_structure) == 26
    assert replaced_structure.get_chemical_symbols() == ["C", "H", "H", "H", "C", "H", "H",
        "C", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "H"]
# TODO: assert positions are the same as when we started

def test_translate_molecule_origin__on_benzene_after_translation_relative_atom_positions_are_unchanged():
    with importlib.resources.path(tests, "benzene.xyz") as molecule_path:
        molecule = ase.io.read(molecule_path)
        molecule_copy = ase.Atoms.copy(molecule)

    translated_molecule = translate_molecule_origin(molecule)
    assert len(translated_molecule) == 9
    assert translated_molecule.get_chemical_symbols() == ["C", "C", "C", "C", "C", "C", "H", "H", "H"]

    for j, position in enumerate(translated_molecule.positions):
        assert position - molecule_copy.positions[j] == approx([-4.68905, -23.36598, -8.48192], 5e-2)

    assert ((translated_molecule[0].position - translated_molecule[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
    assert ((translated_molecule[0].position - translated_molecule[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
    assert ((translated_molecule[0].position - translated_molecule[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
    assert ((translated_molecule[5].position - translated_molecule[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_translate_replace_pattern__on_benzene_after_translation_relative_atom_positions_are_unchanged():
    with importlib.resources.path(tests, "benzene.xyz") as search_path:
        search_pattern = ase.io.read(search_path)
    with importlib.resources.path(tests, "HKUST-1_benzene_replace.xyz") as replace_path:
        replace_pattern = ase.io.read(replace_path)
    molecule_copy = ase.Atoms.copy(replace_pattern)

    translated_molecule = translate_replace_pattern(replace_pattern, search_pattern)
    assert len(translated_molecule) == 9
    assert translated_molecule.get_chemical_symbols() == ["C", "C", "C", "C", "C", "C", "H", "H", "H"]

    for j, position in enumerate(translated_molecule.positions):
        assert position - molecule_copy.positions[j] == approx([1.71202757,1.712034, -9.37916086], 5e-2)

    assert ((translated_molecule[0].position - translated_molecule[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
    assert ((translated_molecule[0].position - translated_molecule[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
    assert ((translated_molecule[0].position - translated_molecule[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
    assert ((translated_molecule[5].position - translated_molecule[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_rotate_replace_pattern__rotate_benzene_90_degrees_about_z_axis_at_origin_changes_x_y_positions():
    with importlib.resources.path(tests, "benzene_at_origin.xyz") as pattern_path:
        pattern = ase.io.read(pattern_path)

    molecule_copy = ase.Atoms.copy(pattern)

    # Rotate the linker by 90 degrees about z axis at origin (0, 0, 0)
    r =  R.from_euler('z', 90, degrees=True)
    rot_linker_euler = r.apply(molecule_copy.positions)

    # Rotate the linker by 90 degrees about z axis at origin (0, 0, 0)
    rotated_pattern = rotate_replace_pattern(pattern, 0, [0, 0, 1], 1.5707963267948966)

    assert len(rotated_pattern) == 9
    assert rotated_pattern.get_chemical_symbols() == ["C", "C", "C", "C", "C", "C", "H", "H", "H"]

    for j, position in enumerate(rotated_pattern.positions):
        assert (position - rot_linker_euler[j]) ** 2 == approx([0,0,0], 5e-2)

    assert ((rotated_pattern[0].position - rotated_pattern[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
    assert ((rotated_pattern[0].position - rotated_pattern[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
    assert ((rotated_pattern[0].position - rotated_pattern[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
    assert ((rotated_pattern[5].position - rotated_pattern[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_rotate_replace_pattern__rotate_benzene_180_degrees_about_x_axis_at_origin_swaps_x_y_positions():
    with importlib.resources.path(tests, "benzene_at_origin.xyz") as pattern_path:
        pattern = ase.io.read(pattern_path)

    molecule_copy = ase.Atoms.copy(pattern)

    # Rotate the linker by 180 degrees about x axis at origin (0, 0, 0)
    r =  R.from_euler('x', 180, degrees=True)
    rot_linker_euler = r.apply(molecule_copy.positions)

    # Rotate the linker by 180 degrees about x axis at origin (0, 0, 0)
    rotated_pattern = rotate_replace_pattern(pattern, 0, [1, 0, 0], 3.1415926535897)

    assert len(rotated_pattern) == 9
    assert rotated_pattern.get_chemical_symbols() == ["C", "C", "C", "C", "C", "C", "H", "H", "H"]

    for j, position in enumerate(rotated_pattern.positions):
        assert (position - rot_linker_euler[j]) ** 2 == approx([0,0,0], 5e-2)

    assert ((rotated_pattern[0].position - rotated_pattern[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
    assert ((rotated_pattern[0].position - rotated_pattern[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
    assert ((rotated_pattern[0].position - rotated_pattern[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
    assert ((rotated_pattern[5].position - rotated_pattern[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_rotate_replace_pattern__rotate_benzene_360_degrees_about_vector_1_1_1_at_origin_does_not_change_positions():
    with importlib.resources.path(tests, "benzene_at_origin.xyz") as pattern_path:
        pattern = ase.io.read(pattern_path)

    molecule_copy = ase.Atoms.copy(pattern)

    # Rotate the pattern by 360 degrees about a vector [1, 1, 1] at origin (0, 0, 0)
    rotated_pattern = rotate_replace_pattern(pattern, 0, [1, 1, 1], 6.2831853071794)

    assert len(rotated_pattern) == 9
    assert rotated_pattern.get_chemical_symbols() == ["C", "C", "C", "C", "C", "C", "H", "H", "H"]

    for j, position in enumerate(rotated_pattern.positions):
        assert (position - molecule_copy.positions[j]) ** 2 == approx([0,0,0], 5e-2)

    assert ((rotated_pattern[0].position - rotated_pattern[1].position) ** 2).sum() == approx(5.8620934418, 5e-2)
    assert ((rotated_pattern[0].position - rotated_pattern[3].position) ** 2).sum() == approx(1.9523164046, 5e-2)
    assert ((rotated_pattern[0].position - rotated_pattern[4].position) ** 2).sum() == approx(7.8072193204, 5e-2)
    assert ((rotated_pattern[5].position - rotated_pattern[8].position) ** 2).sum() == approx(0.8683351588, 5e-2)

def test_replace_pattern_orient__in_hkust1_replacing_benzene_with_benzene_does_not_change_positions():
    with importlib.resources.path(tests, "HKUST-1_withbonds.cif") as hkust1_path:
        structure = ase.io.read(hkust1_path)
    with importlib.resources.path(tests, "benzene.xyz") as pattern_path:
        search_pattern = ase.io.read(pattern_path)

    match_indices, match_atoms = find_pattern_in_structure(structure, search_pattern)


    replace_pattern = []
    for i in range(len(match_atoms)):
        replace_pattern.append(ase.io.read(pattern_path))

    translate_molecule_origin(search_pattern)
    for i in range(len(replace_pattern)):
        translate_molecule_origin(replace_pattern[i])

    for i in range(len(replace_pattern)):
        replace_pattern_orient(search_pattern, replace_pattern[i])

    for i in range(len(replace_pattern)):
        replace_pattern_orient(match_atoms[i], replace_pattern[i])

    for i in range(len(replace_pattern)):
        translate_replace_pattern(replace_pattern[i], match_atoms[i])

    for i in range(len(replace_pattern)):
        diff = match_atoms[i].positions - replace_pattern[i].positions
        for j in range(len(diff)):
            for k in range(3):
                (diff[j][k])**2 == approx(0, abs=1e-3)

        assert len(replace_pattern[i]) == 9
        assert replace_pattern[i].get_chemical_symbols() == ["C", "C", "C", "C", "C", "C", "H", "H", "H"]
