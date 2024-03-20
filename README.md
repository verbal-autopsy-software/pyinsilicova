# pyinsilicova

[![pytest](https://github.com/verbal-autopsy-software/pyinsilicova/actions/workflows/python-package.yml/badge.svg)](https://github.com/verbal-autopsy-software/pyinsilicova/actions)


Python implementation of the InSilicoVA algorithm for assigning causes of death to verbal autopsy data.

This version depends on the C++ library boost [https://www.boost.org/](https://www.boost.org/) and the Python package
pyind ([pybind11 docs](https://pybind11.readthedocs.io/en/latest/)).  To build the package, run `python -m build`.
For windows, the path to the boost library must be added to a System environment variable INCLUDE.


Example run:

```python
from insilicova.api import InSilicoVA
from insilicova.utils import get_vadata

va_data = get_vadata("randomva5")

out = InSilicoVA(data=va_data, data_type="WHO2016")
print(out)
results = out.get_results()
results.get_summary() # prints CSMF
results.get_csmf()    # returns CSMF
```


Example of a quick run (i.e., don't to all of the sampling)

```python
from insilicova.api import InSilicoVA
from insilicova.utils import get_vadata

va_data = get_vadata("randomva5")

out = InSilicoVA(data=va_data, data_type="WHO2016", 
                 n_sim=50, burnin=10, thin=2, auto_length=False)
```
