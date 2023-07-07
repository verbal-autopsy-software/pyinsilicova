# pyinsilicova

[![pytest](https://github.com/verbal-autopsy-software/pyinsilicova/actions/workflows/python-package.yml/badge.svg)](https://github.com/verbal-autopsy-software/pyinsilicova/actions)


Python implementation of the InSilicoVA algorithm for assigning causes of death to verbal autopsy data.


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

out = InSilicoVA(data=va_data, data_type="WHO2016", datacheck=False,
                 n_sim=10, burnin=1, thin=1, auto_length=False)
```
