"""Dataset loading, with every adapter registered on import.

Loaders register themselves through
:func:`geoq.datasets.base.register_dataset`, which means a registry entry
exists only if its module has been imported. Leaving that to the caller made
``load_dataset("bci_iv_2a_lr")`` fail inside the experiment runner while the
same name worked in a notebook where the adapter happened to have been
imported by hand -- a difference that depends on import history rather than on
anything in the configuration.

Importing the adapters here removes that. The cost is nothing: MOABB, MNE and
pyRiemann are imported lazily inside the fetch function, so this module loads
in a bare NumPy and SciPy environment and the registry is populated whether or
not the heavy dependencies are installed. A configuration naming a dataset it
cannot download then fails at download time with a message about the missing
extra, rather than at lookup time with a message about an unknown name.
"""

from __future__ import annotations

#  Imported for the side effect of registering its datasets. The alias names
#  the reason it is here and keeps linters from treating it as unused.
from geoq.datasets import moabb_adapter as _moabb_adapter
from geoq.datasets.base import (
    DATASETS,
    EEGDataset,
    load_dataset,
    make_synthetic_eeg,
    register_dataset,
)

__all__ = [
    "DATASETS",
    "EEGDataset",
    "load_dataset",
    "make_synthetic_eeg",
    "register_dataset",
]

#: Names this module contributed to the registry, for diagnostics.
REGISTERED_BY_IMPORT = tuple(sorted(_moabb_adapter.MOABB_SPECS))
