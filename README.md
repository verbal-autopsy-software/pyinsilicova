# insilicova (Python version)

[![image](https://img.shields.io/pypi/pyversions/insilicova)](https://pypi.org/project/insilicova/)


Python implementation of the InSilicoVA algorithm for assigning causes of death
to verbal autopsy (VA) data collected with the 2016 WHO VA instrument.  This
package is an attempt to replicate the R version
[InSilicoVA](https://github.com/verbal-autopsy-software/InSilicoVA), but the R
version offers more features and functionality via the
[openva](https://github.com/verbal-autopsy-software/openVA) R package.

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

The insilicova package depends on the C++ library [boost](https://www.boost.org/)
(v1.82.0), and the Python package [pyind11](https://github.com/pybind/pybind11)
([pybind11 docs](https://pybind11.readthedocs.io/en/latest/)) is used to help build insilicova.

* On Windows it is assumed that boost is installed at:
  `C:\Program Files\boost\boost_1_82_0` (as specified in `setup.py`)

* Build the package with `python -m build`

