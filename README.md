# pyinsilicova

[![pytest](https://github.com/verbal-autopsy-software/pyinsilicova/actions/workflows/python-package.yml/badge.svg)](https://github.com/verbal-autopsy-software/pyinsilicova/actions)


Python implementation of the InSilicoVA algorithm for assigning causes of death to verbal autopsy (VA) data collected with the 2016 WHO VA instrument.

Example run:

```python
from insilicova.api import InSilicoVA
from insilicova.utils import get_vadata

va_data = get_vadata("randomva5")

out = InSilicoVA(data=va_data)
print(out)
results = out.get_results()
results.get_summary() # prints CSMF
results.get_csmf()    # returns CSMF
```


## Build Dependencies

This package depends on the C++ library boost (v1.82.0) [https://www.boost.org/](https://www.boost.org/) and the Python package
pyind11 ([pybind11 docs](https://pybind11.readthedocs.io/en/latest/)).

* On Windows it is assumed that boost is installed at: `C:\Program Files\boost\boost_1_82_0` (as specified in `setup.py`)

* Build the package with `python -m build`

