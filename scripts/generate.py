import time
import numpy as np
from .utilities import *
from .analyze import *

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

