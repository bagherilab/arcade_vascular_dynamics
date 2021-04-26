# arcade_vascular_dynamics

Note that if you are using Python 3.9, the required version of `networkx` will throw the error: `ImportError: cannot import name 'gcd' from 'fractions'.`
Go to the `networkx/algorithms/dag.py` file in your virtual environment (`venv/lib/python3.9/site-packages/networkx/algorithms/dag.py`) and change the line `from fractions import gcd` to `from math import gcd`.
