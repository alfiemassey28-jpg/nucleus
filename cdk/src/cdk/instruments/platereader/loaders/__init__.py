# To register a new loader:
#   1. Create a new module in this directory (e.g. envision.py)
#   2. Define a class that extends PlateReaderLoader and implements load()
#   3. Decorate it: @plate_loader(name="<name>", extensions=[".<ext>"])
#   4. Import it here and add it to __all__ — the import triggers the decorator,
#      and __all__ exposes it as part of the public API

from .biotek import BiotekLoader

__all__ = ["BiotekLoader"]
