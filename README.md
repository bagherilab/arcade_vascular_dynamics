Supporting code for the article:

> J Yu and N Bagheri. (2021). Modular microenvironment components reproduce vascular dynamics _de novo_ in a multi-scale agent-based model. _Cell Systems_. doi: [10.1016/j.cels.2021.05.007](https://doi.org/10.1016/j.cels.2021.05.007)

## Setup files

The `setups` directory contains all the setup files used for running simulations.
Simulations were run using **[ARCADE v2.3](https://github.com/bagherilab/ARCADE/releases/tag/v2.3)**.

The `ROOT_LAYOUTS.xml` setup file creates checkpoints for the different vascular root layouts that are then loaded for any simulations using those layouts.
The case study `VESSEL_COLLAPSE_stabilized.xml` simulations uses modified code; see supplementary materials for details.

## Simulation data

Raw simulation data and results are available on Mendeley Data:

- `SITE ARCHITECTURE` . [http://dx.doi.org/10.17632/2n3mnkz7yc.1](http://dx.doi.org/10.17632/2n3mnkz7yc.1)
- `ESTIMATED HEMODYNAMICS` . [http://dx.doi.org/10.17632/pgzc9f6kn6.1](http://dx.doi.org/10.17632/pgzc9f6kn6.1)
- `EXACT HEMODYNAMICS` . [http://dx.doi.org/10.17632/8k7f7fcg7t.1](http://dx.doi.org/10.17632/8k7f7fcg7t.1)
- `VASCULAR DAMAGE` . [http://dx.doi.org/10.17632/grtx87d27y.1](http://dx.doi.org/10.17632/grtx87d27y.1)
- `VASCULAR FUNCTION` . [http://dx.doi.org/10.17632/rgtddfyp97.1](http://dx.doi.org/10.17632/rgtddfyp97.1)
- `VESSEL COLLAPSE` . [http://dx.doi.org/10.17632/p8yzkdccg7.1](http://dx.doi.org/10.17632/p8yzkdccg7.1)

## Pipeline notebooks

#### Parse simulation outputs

The **[`parse_simulation_outputs`](parse_simulation_outputs.ipynb)** notebook provides the functions and scripts for parsing simulation files (`.json`) into pickled numpy arrays (`.pkl`).
These parsed results are included with the raw simulation data.

#### Analyze data & results

The **[`analyze_data_results`](analyze_data_results.ipynb)** notebook provides functions and scripts for running basic analysis on simulation data and parsed results.
All resulting `.json` and `.csv` files are provided in the `analysis` directory.

Note that if you are using Python 3.9, the required version of `networkx` will throw the error: `ImportError: cannot import name 'gcd' from 'fractions'.`
Go to the `networkx/algorithms/dag.py` file in your virtual environment (`venv/lib/python3.9/site-packages/networkx/algorithms/dag.py`) and change the line `from fractions import gcd` to `from math import gcd`.

#### Generate figure inputs

The **[`generate_figure_inputs`](generate_figure_inputs.ipynb)** notebook walks through all the steps necessary to generate figure input files from raw data, parsed files, and basic analysis files.
All resulting files are provided in the `analysis` directory.
Refer to figure section in notebook for more details.

To view figures, start a local HTTP server from the root folder, which can be done using Python or PHP:

```bash
$ python3 -m http.server
$ php -S 127.0.0.1:8000
```

Note that the links in the notebook to figures assume the local port 8000; if your server is running on a different port, the links to the figures from the notebook will not work.
Instead, you can navigate to `http://localhost:XXXX/` where `XXXX` is the port number and follow links to the figures.

#### Perform linear regression

The **[`perform_linear_regression`](perform_linear_regression.ipynb)** notebook works through the process of performing linear regression on the different combinations of metrics, properties, and measures.
Regression results are compiled into a single file (included in the `analysis` directory) and used to generate the `linear_regression` figure.

#### Explore case study

The **[`explore_case_study`](explore_case_study.ipynb)** notebook explores the case study `VESSEL_COLLAPSE` simulation set.
Raw simulation data and parsed results are available on Mendeley Data (see above) and compiled analysis files used to generate the `case_study` figure are provided in the `analysis` directory.
