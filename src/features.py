from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from torch import tensor

ATOM_VOCABS = {
    # OGB-style atom categories.
    "atomic_num": list(range(1, 119)) + ["misc"],
    "chirality": [
        int(Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
        int(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW),
        int(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW),
        int(Chem.rdchem.ChiralType.CHI_OTHER),
    ],
    "degree": list(range(0, 11)) + ["misc"],
    "formal_charge": list(range(-5, 6)) + ["misc"],
    "hybridization": [
        int(Chem.rdchem.HybridizationType.SP),
        int(Chem.rdchem.HybridizationType.SP2),
        int(Chem.rdchem.HybridizationType.SP3),
        int(Chem.rdchem.HybridizationType.SP3D),
        int(Chem.rdchem.HybridizationType.SP3D2),
        "misc",
    ],
    "aromatic": [0, 1],
    "implicit_h": list(range(0, 9)) + ["misc"],
    "radical_e": list(range(0, 5)) + ["misc"],
    "in_ring": [0, 1],
}

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]

DESCRIPTOR_FUNCTIONS = {
    "MolWt": Descriptors.MolWt,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "MolLogP": Descriptors.MolLogP,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "NumAromaticRings": Lipinski.NumAromaticRings,
    "NumHDonors": Lipinski.NumHDonors,
    "NumHAcceptors": Lipinski.NumHAcceptors,
    "RingCount": Lipinski.RingCount,
    "HeavyAtomCount": Lipinski.HeavyAtomCount,
}


def _one_hot(value, vocab: List) -> List[float]:
    out = [0.0] * len(vocab)
    try:
        idx = vocab.index(value)
    except ValueError:
        idx = vocab.index("misc") if "misc" in vocab else 0
    out[idx] = 1.0
    return out


def atom_features(atom: Chem.Atom) -> np.ndarray:
    vals = [
        (atom.GetAtomicNum(), ATOM_VOCABS["atomic_num"]),
        (int(atom.GetChiralTag()), ATOM_VOCABS["chirality"]),
        (atom.GetTotalDegree(), ATOM_VOCABS["degree"]),
        (atom.GetFormalCharge(), ATOM_VOCABS["formal_charge"]),
        (int(atom.GetHybridization()), ATOM_VOCABS["hybridization"]),
        (int(atom.GetIsAromatic()), ATOM_VOCABS["aromatic"]),
        (atom.GetTotalNumHs(includeNeighbors=True), ATOM_VOCABS["implicit_h"]),
        (atom.GetNumRadicalElectrons(), ATOM_VOCABS["radical_e"]),
        (int(atom.IsInRing()), ATOM_VOCABS["in_ring"]),
    ]
    feat = sum((_one_hot(v, vocab) for v, vocab in vals), [])
    feat.append(0.0)  # Reserved channel for the 174D input.
    return np.asarray(feat, dtype=np.float32)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    return np.asarray(
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + _one_hot(bond.GetStereo(), BOND_STEREO)
        + _one_hot(int(bond.GetIsConjugated()), [0, 1]),
        dtype=np.float32,
    )


def molecule_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = np.stack([atom_features(a) for a in mol.GetAtoms()])
    edges, attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edges.extend([[i, j], [j, i]])
        attrs.extend([feat, feat])

    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_attr = np.stack(attrs)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, len(BOND_TYPES) + 1 + len(BOND_STEREO) + 1 + 3), dtype=np.float32)
    return x, edge_index, edge_attr


def compute_descriptors(smiles: str, names: List[str]) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return {name: float(DESCRIPTOR_FUNCTIONS[name](mol)) for name in names}


def node_feature_dim() -> int:
    return sum(len(v) for v in ATOM_VOCABS.values()) + 1


def edge_feature_dim() -> int:
    return len(BOND_TYPES) + len(BOND_STEREO) + 2
