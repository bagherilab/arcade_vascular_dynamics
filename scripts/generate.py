import time
import numpy as np
from .utilities import *
from .analyze import *
from collections import Counter
from math import sqrt
from scipy.stats import iqr

# GENERAL ANALYSIS FUNCTIONS ===================================================

def merge_seeds(file, out, keys, extension, code, tar=None):
    """Merge seed files across conditions."""
    filepath = f"{file}{code}{extension}.json"

    if tar:
        D = load_json(filepath.split("/")[-1], tar=tar)
    else:
        D = load_json(filepath)

    keys.pop('time', None)
    metric = extension.split(".")[-1]

    if metric == "POPS" or metric == "TYPES":
        timepoints = []

        for i, d in enumerate(D):
            for key in keys.keys():
                for dd in d:
                    dd[key] = keys[key]

            if i == 0:
                for dd in d:
                    dd['_'] = [dd['_']]
                timepoints = timepoints + d
            else:
                for tp, dd in zip(timepoints, d):
                    tp['_'].append(dd['_'])
                pass

        out['data'] = out['data'] + timepoints
    else:
        for key in keys.keys():
            for d in D:
                d[key] = keys[key]
        out['data'] = out['data'] + D

def merge_centers(file, out, keys, extension, code, tar=None):
    """Merge center concentrations across conditions."""
    code = code.replace("_CHX_", "_CH_")
    filepath = f"{file}{code}{extension}.json"

    if tar:
        D = load_json(filepath.split("/")[-1], tar=tar)
    else:
        D = load_json(filepath)

    keys['glucose'] = D['glucose']
    keys['oxygen'] = D['oxygen']

    keys.pop('time', None)
    out['data'].append(keys)

    if "_X" in D:
        out['_X'] = D['_X']

def merge_graph(file, out, keys, extension, code, tar=None):
    """Merge graph files across conditions."""
    code = code.replace("_CHX_", "_CH_")
    filepath = f"{file}{code}.GRAPH.{keys['time']}.csv"
    metric = extension.split(".")[-1]
    print(code, keys['time'])

    if tar:
        D = load_csv(filepath.split("/")[-1], tar=tar)
    else:
        D = load_csv(filepath)

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    header = D[0]
    d = D[1:]

    ind_seed = header.index('seed')
    ind_fromx = header.index('fromx')
    ind_tox = header.index('tox')
    ind_fromy = header.index('fromy')
    ind_toy = header.index('toy')
    ind_flow = header.index('FLOW')

    time = unformat_time(keys['time'])
    d_time = np.array([[float(e) for e in edge] for edge in d if edge[-1] != "nan"])

    all_means = np.empty((10,18))
    all_stds = np.empty((10,18))
    all_mins = np.empty((10,18))
    all_maxs = np.empty((10,18))
    all_medians = np.empty((10,18))
    all_uppers = np.empty((10,18))
    all_lowers = np.empty((10,18))
    all_totals = np.empty(10)
    all_density = np.empty(10)
    all_perfusion = np.empty(10)

    all_means[:] = np.nan
    all_stds[:] = np.nan
    all_mins[:] = np.nan
    all_maxs[:] = np.nan
    all_medians[:] = np.nan
    all_uppers[:] = np.nan
    all_lowers[:] = np.nan
    all_totals[:] = np.nan
    all_density[:] = np.nan
    all_perfusion[:] = np.nan

    ind_pressure_from = header.index('frompressure')
    ind_pressure_to = header.index('topressure')

    # simulation size
    radius = 40
    length = 4*radius - 2
    width = (6*radius - 3 + 1)/2
    depth = 8.7
    side = 15

    if len(d_time) != 0:
        for i, seed in enumerate(seeds):
            d_time_seed = d_time[d_time[:,ind_seed] == int(seed),:]
            if len(d_time_seed) == 0:
                continue

            all_means[i,:] = np.mean(d_time_seed, axis=0)
            all_stds[i,:] = np.std(d_time_seed, axis=0, ddof=1)
            all_mins[i,:] = np.min(d_time_seed, axis=0)
            all_maxs[i,:] = np.max(d_time_seed, axis=0)

            all_medians[i,:] = np.percentile(d_time_seed, 50, axis=0)
            all_lowers[i,:] = np.percentile(d_time_seed, 25, axis=0)
            all_uppers[i,:] = np.percentile(d_time_seed, 75, axis=0)

            all_totals[i] = d_time_seed.shape[0]

            # calculate capillary density
            coords = d_time_seed[:,[ind_fromx,ind_tox]]
            slices = [i for fromx, tox in coords for i in range(int(min(tox,fromx)) + 1, int(max(tox, fromx)))]
            counts = dict(Counter(slices))
            area = length*side*depth*2 # slice is two layers, um^2
            count_values = [counts[k]/area*1E6 for k in counts] # units of cap/mm^2
            all_density[i] = np.mean(count_values)

            # calculate perfusion
            mean_flow = all_means[i,ind_flow]/60 # um^3/sec
            volume = (length*side)*(width*2*side/sqrt(3))*(depth*2) # um^3
            all_perfusion[i] = mean_flow/volume

    if metric == "NUMBER":
        keys["_"] = { "mean" : all_totals.tolist() }
    elif metric == "DENSITY":
        keys["_"] = { "mean" : all_density.tolist() }
    elif metric == "PERFUSION":
        keys["_"] = { "mean" : all_perfusion.tolist() }
    elif metric == "PRESSURE":
        keys["_"] = {
            "mean": np.divide(np.add(all_means[:,ind_pressure_from], all_means[:,ind_pressure_to]), 2).tolist(),
            "std": np.divide(np.add(all_stds[:,ind_pressure_from], all_stds[:,ind_pressure_to]), 2).tolist(),
            "min": np.divide(np.add(all_mins[:,ind_pressure_from], all_mins[:,ind_pressure_to]), 2).tolist(),
            "max": np.divide(np.add(all_maxs[:,ind_pressure_from], all_maxs[:,ind_pressure_to]), 2).tolist(),
            "median": np.divide(np.add(all_medians[:,ind_pressure_from], all_medians[:,ind_pressure_to]), 2).tolist(),
            "lower": np.divide(np.add(all_lowers[:,ind_pressure_from], all_lowers[:,ind_pressure_to]), 2).tolist(),
            "upper": np.divide(np.add(all_uppers[:,ind_pressure_from], all_uppers[:,ind_pressure_to]), 2).tolist(),
        };
    elif metric == "OXYGEN":
        ind_oxygen_from = header.index('fromoxygen')
        ind_oxygen_to = header.index('tooxygen')
        keys["_"] = {
            "mean": np.divide(np.add(all_means[:,ind_oxygen_from], all_means[:,ind_oxygen_to]), 2).tolist(),
            "std": np.divide(np.add(all_stds[:,ind_oxygen_from], all_stds[:,ind_oxygen_to]), 2).tolist(),
            "min": np.divide(np.add(all_mins[:,ind_oxygen_from], all_mins[:,ind_oxygen_to]), 2).tolist(),
            "max": np.divide(np.add(all_maxs[:,ind_oxygen_from], all_maxs[:,ind_oxygen_to]), 2).tolist(),
            "median": np.divide(np.add(all_medians[:,ind_oxygen_from], all_medians[:,ind_oxygen_to]), 2).tolist(),
            "lower": np.divide(np.add(all_lowers[:,ind_oxygen_from], all_lowers[:,ind_oxygen_to]), 2).tolist(),
            "upper": np.divide(np.add(all_uppers[:,ind_oxygen_from], all_uppers[:,ind_oxygen_to]), 2).tolist(),
        };
    else:
        ind = header.index(metric)
        keys["_"] = {
            "mean": all_means[:,ind].tolist(),
            "std": all_stds[:,ind].tolist(),
            "min": all_mins[:,ind].tolist(),
            "max": all_maxs[:,ind].tolist(),
            "median": all_medians[:,ind].tolist(),
            "lower": all_lowers[:,ind].tolist(),
            "upper": all_uppers[:,ind].tolist(),
        };

    out['data'].append(keys)

# ------------------------------------------------------------------------------

def save_seeds(file, extension, out):
    """Save merged seed files."""
    save_json(file, out, extension)

def save_centers(file, extension, out):
    """Save merged center concentrations."""
    save_json(file, out, extension)

def save_graph(file, extension, out):
    """Save merged graph files."""
    save_json(file, out['data'], extension)

# POINT DISTANCES ==============================================================

def get_type_index(type):
    """Denote cell type as active (0) or inactive (1)."""
    return 0 if type in [2, 3, 4] else 1

def get_max_distance(arr):
    """Get maximum non-zero distance."""
    non_zero = [ind for ind, value in enumerate(arr) if value > 0]
    return non_zero[-1] + 1 if len(non_zero) > 0 else 0

def make_point_distances(D, R, H, T, N, C, POPS, TYPES, outfile, code, exclude=[-1], timepoints=[], seeds=[]):
    """Extract point distances."""
    arrt = np.zeros((N, R, 2), dtype=np.int32)
    type_index_names = ['active', 'inactive']

    start = time.time()
    [np.add.at(arrt, (i, get_radius(C[m]), get_type_index(e)), 1)
        for i, seed in enumerate(D['agents']['type'])
        for m, loc in enumerate(seed[-1][0])
        for p, e in enumerate(loc)
            if D['agents']['pop'][i,-1,0,m,p] not in exclude]
    end = time.time()
    print(end - start)

    indices = [[get_max_distance(arrt[seed,:,i]) for i in [0, 1]] for seed in range(0,N)]
    point = int(code.split("_")[-1].replace("point", ""))

    out = [[point, seed, type_index_names[i], indices[seed][i]]
        for seed in range(0, N)
        for i in [0, 1]]

    header = "point,seed,group,index\n"
    save_csv(f"{outfile}{code}", header, zip(*out), ".DISTANCES")

def merge_point_distances(file, out, keys, extension, code, tar=None):
    """Merge point distance files across conditions."""
    filepath = f"{file}{code}{extension}.csv"

    if tar:
        D = load_csv(filepath.split("/")[-1], tar=tar)
    else:
        D = load_csv(filepath)

    d = [[keys['context']] + e for e in D[1:]]
    out['data'] = out['data'] + d
    out['header'] = ["context"] + D[0]

def save_point_distances(file, extension, out):
    """Save merged point distances file."""
    save_csv(file, ','.join(out['header']) + "\n", zip(*out['data']), extension)

# TYPE GRID ====================================================================

def make_type_grid_borders(D, R, H, T, N, C, POPS, TYPES, outfile, code, exclude=[-1], timepoints=[], seeds=[]):
    """Extract type grid borders."""
    d = np.take(D["agents"], timepoints, axis=1)
    TT = [T[i] for i in timepoints]

    inds = [[get_inds(d, j, i, H, exclude)
        for j in range(0, N)]
        for i in range(0, len(TT))]

    all_inds = [x for ind in inds[0] for x in ind]
    lines = lin = get_spatial_outlines(C, 1, R, H, [all_inds])
    outline = []
    xy, offx, offy, L, W = convert(C, R)

    for z, li in zip(range(1 - H, H), lin):
        outline = outline + [[i - offx, j - offy, z, k, C]
            for i, A in enumerate(li)
            for j, B in enumerate(A)
            for k, C in enumerate(B) if C != 0]

    header = "x,y,z,DIRECTION,WEIGHT\n"
    save_csv(f"{outfile}{code}", header, zip(*outline), f".BORDERS.{format_time(TT[0])}")

def merge_type_grid_borders(file, out, keys, extension, code, tar=None):
    """Merge type grid border files across conditions."""
    filepath = f"{file}{code}{extension}.150.csv"

    if tar:
        D = load_csv(filepath.split("/")[-1], tar=tar)
    else:
        D = load_csv(filepath)

    d = [[keys['context'], keys['sites'].replace("SOURCE_", "")] + e for e in D[1:]]
    out['data'] = out['data'] + d
    out['header'] = ["context", "grid"] + D[0]

def save_type_grid_borders(file, extension, out):
    """Save merged type grid border files."""
    save_csv(file, ','.join(out['header']) + "\n", zip(*out['data']), extension)

# PATTERN COMPARE ==============================================================

def make_pattern_compare_metrics(file, metric, names, contexts, cases):
    """Compiles selected metrics for pattern layout."""
    out = {}

    for name in names:
        outfile = f"{file}{name}/{name}"
        extension = f".METRICS.{metric}"
        tar = load_tar(outfile, extension)

        selected = []

        for context in contexts:
            for case in cases[name]:
                code = "_".join([value for key, value in case])
                filepath = f"{outfile}_{context}_{code}{extension}.json"

                # Try loading from tar.
                if tar:
                    D = load_json(filepath.split("/")[-1], tar=tar)
                else:
                    D = load_json(filepath)

                keys = {key: value for key, value in case}
                keys["_Y"] = D["mean"]
                keys["context"] = context

                selected.append(keys)

        out["_X"] = D["_X"]
        out[name] = selected

    save_json(f"{file}_/PATTERN_COMPARE", out, f".{metric}")

def make_pattern_compare_concentrations(input_path, output_path, names, contexts, cases):
    """Compiles selected metrics for pattern layout."""
    out = {
        "glucose": { }, "oxygen": { }
    }

    for name in names:
        outfile = f"{output_path}{name}/{name}"
        out["glucose"][name] = []
        out["oxygen"][name] = []

        for context in contexts:
            for case in cases[name]:
                add = "damage" if name == "VASCULAR_DAMAGE" else ""
                code = "_".join([value for key, value in case])
                infile = f"{input_path}{name}/{name}_{context}_{add}{code}.tar.xz"
                tar = tarfile.open(infile)

                keys = {key: value for key, value in case}
                keys["context"] = context
                get_pattern_compare_concentrations(tar, out, keys, name)

    glucose = out["glucose"]
    glucose["_X"] = out["_X"]

    oxygen = out["oxygen"]
    oxygen["_X"] = out["_X"]

    save_json(f"{output_path}_/PATTERN_COMPARE", glucose, f".GLUCOSE")
    save_json(f"{output_path}_/PATTERN_COMPARE", oxygen, f".OXYGEN")

def get_pattern_compare_concentrations(tar, out, keys, name):
    """Extract center concentration over time."""

    concentrations = ["glucose", "oxygen"]
    arr = np.zeros((31, len(concentrations)))

    seeds = 0
    for member in tar.getmembers():
        seed = int(re.findall(r'_([0-9]{2})\.json', member.name)[0])
        json = load_json(member, tar=tar)

        for t, jn in enumerate(json["timepoints"]):
            for c, conc in enumerate(concentrations):
                concs = jn['molecules'][conc]

                if len(concs) > 1:
                    mid = int((len(concs) - 1)/2)
                    cc = concs[mid]
                    arr[t,c] += cc[0]
                else:
                    arr[t,c] += concs[0][0]

        seeds = seeds + 1

    out['_X'] = [x['time'] for x in json["timepoints"]]
    for c, conc in enumerate(concentrations):
        entry = dict(keys)
        entry["_Y"] = np.divide(arr[:,c], seeds).tolist()
        out[conc][name].append(entry)

# PATTERN TYPES ================================================================

def make_pattern_types_locations(file, names, contexts, cases):
    """Compiles locations for pattern types."""
    out = []

    for name in names:
        outfile = f"{file}{name}/{name}"
        tar = load_tar(outfile, ".LOCATIONS")

        for context in contexts:
            for case in cases[name]:
                code, a, b = case
                filepath = f"{outfile}_{context}_{code}.LOCATIONS.150.csv"

                # Try loading from tar.
                if tar:
                    D = load_csv(filepath.split("/")[-1], tar=tar)
                else:
                    D = load_csv(filepath)

                if context == "C":
                    contents = [[context, a, b] + row[:6] + ["nan"] + row[6:] for row in D[1:]]
                else:
                    contents = [[context, a, b] + row for row in D[1:]]
                out = out + contents

    header = ["context", "dynamics", "coupling"] + D[0]
    save_csv(f"{file}_/PATTERN_TYPES", ','.join(header) + "\n", zip(*out), ".LOCATIONS")

def make_pattern_types_borders(input_path, output_path, names, contexts, cases):
    """Compiles borders for pattern types."""
    out = []

    for name in names:
        outfile = f"{input_path}{name}/{name}"

        for context in contexts:
            exclude = [-1] if context == "C" else [-1, 1]

            for case in cases[name]:
                code, a, b = case
                add = "damage" if name == "VASCULAR_DAMAGE" else ""
                filepath = f"{outfile}_{context.replace('CHX','CH')}_{add}{code}.pkl"

                D, R, H, T, N, C, POPS, TYPES = load(filepath)

                dd = np.take(D["agents"], [30], axis=1)
                TT = [T[i] for i in [30]]

                inds = [[get_inds(dd, j, i, H, exclude)
                    for j in range(0, N)]
                    for i in range(0, len(TT))]

                all_inds = [x for ind in inds[0] for x in ind]
                lines = lin = get_spatial_outlines(C, 1, R, H, [all_inds])
                outline = []
                xy, offx, offy, L, W = convert(C, R)

                for z, li in zip(range(1 - H, H), lin):
                    outline = outline + [[context, a, b, i - offx, j - offy, z, k, C]
                        for i, A in enumerate(li)
                        for j, B in enumerate(A)
                        for k, C in enumerate(B) if C != 0]

                out = out + outline

    header = "context,dynamics,coupling,x,y,z,DIRECTION,WEIGHT\n"
    save_csv(f"{output_path}_/PATTERN_TYPES", header, zip(*out), ".BORDERS")

# LAYOUT MERGED ================================================================

def make_layout_merged_metrics(D, R, H, T, N, C, POPS, TYPES, outfile, code, exclude=[-1], timepoints=[], seeds=[]):
    """Compile emergent metrics for different layouts."""
    start = time.time()
    inds = [[get_inds(D["agents"], j, i, H, exclude)
        for i in range(0, len(T))]
        for j in range(0, N)]
    end = time.time()
    print(end - start)

    cycles = get_temporal_cycles(D["agents"], T, N, inds)

    offset = 4
    growth = get_temporal_growths(T, N, C, inds, offset)
    nan_growth = [np.nan] * (offset + 2)

    symmetry = get_temporal_symmetries(T, N, C, R, inds)

    activity = get_temporal_activity(D["agents"], T, N, inds, TYPES)

    out = {
        "cycles": cycles,
        "growth": [nan_growth + grow for grow in growth],
        "symmetry": symmetry,
        "activity": activity.tolist()
    }

    save_json(f"{outfile}{code}", out, ".MERGED.METRICS")

def make_layout_merged_concentrations(tar, timepoints, keys, outfile, code):
    """Compile center concentrations for different layouts."""
    seeds = 10
    out = {}
    arr = np.zeros((seeds, 31, 2))

    i = 0
    for member in tar.getmembers():
        seed = int(re.findall(r'_([0-9]{2})\.json', member.name)[0])
        json = load_json(member, tar=tar)

        for t, timepoint in enumerate(json['timepoints']):
            glucose = timepoint['molecules']['glucose'][0][0]
            oxygen = timepoint['molecules']['oxygen'][0][0]

            arr[i,t,0] = glucose
            arr[i,t,1] = oxygen

        i = i + 1

    out = {
        "glucose": arr[:,:,0].tolist(),
        "oxygen": arr[:,:,1].tolist()
    }

    save_json(f"{outfile}{code}", out, ".MERGED.CONCENTRATIONS")

def make_layout_merged(file, metric):
    """Merge layout files for given metric."""
    out = {}

    codes = [
        ("PATTERN", "pattern"),
        ("Lav", "graphs"),
        ("Lava", "graphs"),
        ("Lvav", "graphs"),
        ("Sav", "graphs"),
        ("Savav", "graphs"),
    ]

    for name in ["EXACT_HEMODYNAMICS", "VASCULAR_FUNCTION"]:
        outfile = f"{file}{name}/{name}"
        out[name] = {}

        for context in ["C", "CHX"]:
            tar = load_tar(outfile, ".MERGED")

            contents = {}
            contents["pattern"] = []
            contents["graphs"] = []

            for code, cat in codes:
                extension = "CONCENTRATIONS" if metric in ["GLUCOSE", "OXYGEN"] else "METRICS"
                file_context = context.replace("CHX", "CH") if metric in ["GLUCOSE", "OXYGEN"] else context
                filepath = f"{outfile}_{file_context}_{code}.MERGED.{extension}.json"

                if tar:
                    D = load_json(filepath.split("/")[-1], tar=tar)
                else:
                    D = load_json(filepath)

                contents[cat] = contents[cat] + D[metric.lower()]

            # replace nan with np.nan
            pattern = np.array([[np.nan if tp == "nan" else tp for tp in seed] for seed in contents["pattern"]])
            graphs = np.array([[np.nan if tp == "nan" else tp for tp in seed] for seed in contents["graphs"]])

            pattern_mean = np.mean(pattern, axis=0)
            pattern_std = np.std(pattern, axis=0, ddof=1)
            pattern_int = 2.262*pattern_std/sqrt(len(pattern))

            graphs_mean = np.mean(graphs, axis=0)
            graphs_std = np.std(graphs, axis=0, ddof=1)
            graphs_int = 2.262*graphs_std/sqrt(len(graphs))

            out[name][context] = {
                "pattern": {
                    "mean": pattern_mean.tolist(),
                    "std": pattern_std.tolist(),
                    "int": pattern_int.tolist()
                },
                "graphs": {
                    "mean": graphs_mean.tolist(),
                    "std": graphs_std.tolist(),
                    "int": graphs_int.tolist()
                }
            }

    out["_T"] = list(np.arange(0, 15.5, 0.5))
    save_json(f"{file}_/LAYOUT_MERGED", out, f".{metric}")

# LAYOUT SCATTER ===============================================================

def make_layout_scatter(file):
    """Merge layout graph files for scatter."""
    out = {}

    codes = [
        ("PATTERN", "pattern"),
        ("Lav", "graphs"),
        ("Lava", "graphs"),
        ("Lvav", "graphs"),
        ("Sav", "graphs"),
        ("Savav", "graphs"),
    ]

    for name, couple in [("EXACT_HEMODYNAMICS", "UNCOUPLED"), ("VASCULAR_FUNCTION", "COUPLED")]:
        outfile = f"{file}{name}/{name}"
        tar = load_tar(outfile, ".GRAPH")

        out = {}
        out["pattern"] = []
        out["graphs"] = []

        for context in ["C", "CH"]:
            for code, cat in codes:
                filepath = f"{outfile}_{context}_{code}.GRAPH.150.csv"

                if tar:
                    D = load_csv(filepath.split("/")[-1], tar=tar)
                else:
                    D = load_csv(filepath)

                header = D[0]
                contents = [[context] + row[4:6] + row[9:] for row in D[1:]]
                out[cat] = out[cat] + contents

        full_header = ",".join(["context"] + header[4:6] + header[9:]) + "\n"
        save_csv(f"{file}_/LAYOUT_SCATTER", full_header, zip(*out["graphs"]), f".ROOT.{couple}")
        save_csv(f"{file}_/LAYOUT_SCATTER", full_header, zip(*out["pattern"]), f".PATTERN.{couple}")

# PROPERTY DISTRIBUTION ========================================================

def bin_property_values(data, name, lower, upper):
    """Bins data between given bounds."""
    filtered_data = [d for d in data if not np.isnan(d) and not np.isinf(d)]

    if len(filtered_data) == 0:
        return { "hist": [], "bins": [] }

    data_iqr = iqr(filtered_data)
    bandwidth = 2*data_iqr/(len(filtered_data)**(1./3.))

    min_value = np.min(filtered_data)
    max_value = np.max(filtered_data)

    if min_value < lower:
        lower = min_value*0.5
        print(f"\t\t> Changed lower bound for {name} to {lower}")

    if max_value > upper:
        upper = max_value*1.5
        print(f"\t\t> Change upper bound for {name} to {upper}")

    print(f"{name:>10} = [{min_value}, {max_value}]")

    hist, bins = np.histogram(filtered_data, bins=np.arange(lower, upper, bandwidth))
    return { "hist": hist.tolist(), "bins": bins.tolist() }

def make_property_distribution(file):
    """Bin and merge graph properties."""
    codes = ["PATTERN", "Lav", "Lava", "Lvav", "Sav", "Savav"]
    properties = ["RADIUS", "PRESSURE", "WALL", "SHEAR", "FLOW", "CIRCUM"]
    names = [
        ("EXACT_HEMODYNAMICS", "C", "C/CH"),
        ("VASCULAR_FUNCTION", "C", "C"),
        ("VASCULAR_FUNCTION", "CH", "CH"),
    ]
    out = { prop: {} for prop in properties }

    for name, context, context_code in names:
        outfile = f"{file}{name}/{name}"
        tar = load_tar(outfile, ".GRAPH")

        for graph_code in codes:
            filepath = f"{outfile}_{context}_{graph_code}.GRAPH.150.csv"

            if tar:
                D = load_csv(filepath.split("/")[-1], tar=tar)
            else:
                D = load_csv(filepath)

            header = D[0]
            ind_pressure_from = header.index('frompressure')
            ind_pressure_to = header.index('topressure')
            ind_radius = header.index("RADIUS")
            ind_wall = header.index("WALL")
            ind_shear = header.index("SHEAR")
            ind_flow = header.index("FLOW")
            ind_circum = header.index("CIRCUM")

            pressure = [(float(dd[ind_pressure_to]) + float(dd[ind_pressure_from]))/2*0.133322 for dd in D[1:]] # mmHg -> kPa
            radius = [float(dd[ind_radius]) for dd in D[1:]]
            wall = [float(dd[ind_wall]) for dd in D[1:]]
            shear = [np.log10(float(dd[ind_shear])*133.322) for dd in D[1:]] # mmHg -> Pa
            flow = [np.log10(float(dd[ind_flow])) for dd in D[1:]]
            circum = [np.log10(float(dd[ind_circum])*133.322) for dd in D[1:]] # mmHg -> Pa

            key = f"{context_code}_{graph_code}"
            out["RADIUS"][key] = bin_property_values(radius, "RADIUS", 0, 50)
            out["PRESSURE"][key] = bin_property_values(pressure, "PRESSURE", 0, 10)
            out["WALL"][key] = bin_property_values(wall, "WALL", 0, 10)
            out["SHEAR"][key] = bin_property_values(shear, "SHEAR", -4, 3)
            out["FLOW"][key] = bin_property_values(flow, "FLOW", 2, 12)
            out["CIRCUM"][key] = bin_property_values(circum, "CIRCUM", 3, 6)

    for prop in properties:
        save_json(f"{file}_/PROPERTY_DISTRIBUTION", out[prop], f".{prop}")

# GRAPH MEASURES ===============================================================

def make_graph_measures(file):
    """Extract graph measures."""
    codes = ["Lav", "Lava", "Lvav", "Sav", "Savav"]
    names = [
        ("EXACT_HEMODYNAMICS", "C", "C/CH"),
        ("VASCULAR_FUNCTION", "C", "C"),
        ("VASCULAR_FUNCTION", "CH", "CH"),
    ]
    out = []

    for name, context, context_code in names:
        outfile = f"{file}{name}/{name}"
        tar = load_tar(outfile, ".MEASURES")

        for graph_code in codes:
            filepath = f"{outfile}_{context}_{graph_code}.MEASURES.csv"

            if tar:
                D = load_csv(filepath.split("/")[-1], tar=tar)
            else:
                D = load_csv(filepath)

            header = D[0]
            content = [[context_code, graph_code] + row[2:] for row in D[1:] if row[0] == "150"]
            out = out + content

    full_header = ",".join(["context", "graph"] + header[2:]) + "\n"
    save_csv(f"{file}_/GRAPH_MEASURES", full_header, zip(*out), "")

# EDGE COUNTS ==================================================================

def make_edge_counts(tar, timepoints, keys, outfile, code):
    """Count number of edges over time."""
    seeds = 10
    arr = np.zeros((10, 31))

    for member in tar.getmembers():
        seed = int(re.findall(r'_([0-9]{2})\.GRAPH\.json', member.name)[0])
        json = load_json(member, tar=tar)
        timepoints = json['timepoints']
        t = [tp['time'] for tp in timepoints]
        e = [len(tp['graph']) for tp in timepoints]
        arr[seed,:] = e

    starting = arr[:,0]
    relative = (starting[:,None] - arr)/(starting[:,None])

    out = {
        "time": t,
        "_": {
            "seeds": arr.tolist(),
            "mean": np.mean(relative, axis=0).tolist(),
            "std":  np.std(relative, axis=0, ddof=1).tolist(),
            "min":  np.min(relative, axis=0).tolist(),
            "max":  np.max(relative, axis=0).tolist(),
        }
    }

    save_json(f"{outfile}{code}", out, ".EDGES")

def merge_edge_counts(file, out, keys, extension, code, tar=None):
    """Merge edge count files across conditions."""
    filepath = f"{file}{code.replace('CHX', 'CH')}{extension}.json"

    if tar:
        D = load_json(filepath.split("/")[-1], tar=tar)
    else:
        D = load_json(filepath)

    keys["_"] = D["_"]
    keys.pop('time', None)
    out['data'].append(keys)
    out['time'] = D['time']

def save_edge_counts(file, extension, out):
    """Save merged edge counts file."""
    save_json(file, out, extension)
