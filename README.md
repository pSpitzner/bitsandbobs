# Bits and Bobs

This is a collection of little python hacks and workarounds that I use frequently across projects.


## Install

Some dependencies that I recommend installing via conda:
```bash
conda install numpy matplotlib h5py
```

Our stuff:
```bash
python -m pip install git+https://github.com/pSpitzner/bitsandbobs
```

Then, in python:
```python
import bitsandbobs as bnb
```

[Why not simply `pip install`?](https://adamj.eu/tech/2020/02/25/use-python-m-pip-everywhere/)


## Features

Just the noteworthy ones. See each functions docstring for details.

Load and write a hdf5 files as/from nested dictionaries.
```
h5f = bnb.hi5.recursive_load("~/path/to/in_file.h5")
bnb.hi5.recursive_write("~/path/to/out_file.h5", h5f)
```
---

Set the size of an existing matplotlib axes element (and the padding around it).
Useful when creating multi-panel figures in Affinity Designer or Inkscape.

```
fig, ax = plt.subplots()
bnb.plt.set_size(ax, w=3, h=2)
```

---

Convert transparent colors (r,g,b,a) to opaque equivalents (r,g,b) on a given background.
```
bnb.plt.alpha_to_solid_on_bg(base="red", alpha=0.5, bg="white")
```

