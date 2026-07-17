"""
Module for loading data from a microplate reader.

Loading data
------------
Use load_platereader_data() to load a file — the loader is chosen automatically
from the file extension:

    # TODO: add example here

Pass platereader= to override the loader explicitly:

    # TODO: add example here

Adding a new loader
-------------------
PlateReaderLoader subclasses are registered via the @plate_loader decorator,
which maps a loader name and file extensions to the implementing class.

    1. Create a new module under loaders/ (e.g. loaders/envision.py)
    2. Subclass PlateReaderLoader and implement load()
    3. Decorate the class:
         @plate_loader(name="envision", extensions=[".csv"])
         class EnvisionLoader(PlateReaderLoader): ...
    4. Import the class in loaders/__init__.py and add it to __all__
       The import is what triggers the decorator and populates the registry.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from .result import PlateReaderResult
from .base import PlateReaderLoader, plate_loader, _REGISTRY, _EXT_REGISTRY
from ..platemap import load_platemap
from . import loaders as _loaders  # noqa: F401 — triggers @plate_loader registrations
from .constants import PlateReaderDataColumn

# TODO: add normalization directly into this function??
def load_platereader_data(
    data_file: Path | str,
    platemap_file: Path | str | None = None,
    platereader: str | None = None,
) -> PlateReaderResult:
    """Load a plate reader file, auto-detecting the loader from the file extension.

    Args:
        data_file: Path to the plate reader data file.
        platemap_file: Optional path to a platemap file. When provided the platemap
            is loaded, validated, and merged into each result's data on ``Well``.
            The platemap is also stored on each returned ``PlateReaderResult``.
        platereader: Optional loader name override (e.g. ``"biotek"``). If omitted,
            the loader is chosen from the data file extension.
    """
    data_file = Path(data_file)
    if platereader is not None:
        if platereader not in _REGISTRY:
            raise ValueError(f"Unknown plate reader loader {platereader!r}. Available: {list(_REGISTRY)}")
        result = _REGISTRY[platereader].load(data_file)
    else:
        ext = data_file.suffix.lower()
        name = _EXT_REGISTRY.get(ext)
        if name is None:
            raise ValueError( 
                f"No plate reader loader registered for extension {ext!r}. "
                f"Available: {list(_EXT_REGISTRY)}. "
                f"Pass platereader= to specify one explicitly."
            )
        result = _REGISTRY[name].load(data_file)

    if platemap_file is not None:
        platemap = load_platemap(platemap_file)
        result = PlateReaderResult(
            data=[dataclasses.replace(r, data=r.data.merge(platemap, on=PlateReaderDataColumn.WELL), platemap_columns=set(platemap.columns)) for r in result.data],
            _plates=result._plates,
            metadata=result.metadata,
            source_file=result.source_file,
            platemap=platemap,
        )


    return result


