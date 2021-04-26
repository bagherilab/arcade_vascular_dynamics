import time
import numpy as np
from .utilities import *
from .analyze import *
from collections import Counter
from math import sqrt

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

# SITE ARCHITECTURE: POINT DISTANCES ===========================================

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

# SITE ARCHITECTURE: TYPE GRID =================================================

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

