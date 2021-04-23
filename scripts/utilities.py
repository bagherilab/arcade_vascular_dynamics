import pickle

def load(filename):
    """Load contents of parsed results file."""
    D = pickle.load(open(filename, "rb"))
    R = D['setup']['radius']
    H = D['setup']['height']
    T = D['setup']['time']
    N = D['agents'].shape[0]
    C = D['setup']['coords']
    POPS = D['setup']['pops']
    TYPES = D['setup']['types']
    return D, R, H, T, N, C, POPS, TYPES
