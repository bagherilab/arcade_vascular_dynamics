import pickle
import json
import csv
import re

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

def load_tar(tar, member):
    """Load member of .tar.xz. file."""
    file = tar.extractfile(member)
    contents = [line.decode("utf-8") for line in file.readlines()]
    return json.loads("".join(contents))

def load_json(json_file):
    """Load .json file."""
    return json.load(open(json_file, "r"))

def save_json(filename, out, extension):
    """Save contents as json."""
    with open(filename + extension + ".json", "w") as f:
        jn = json.dumps(out, indent = 4, separators = (',', ':'), sort_keys=True)
        f.write(format_json(jn).replace("NaN", '"nan"'))

def save_csv(filename, header, elements, extension):
    """Save contents as csv."""
    with open(filename + extension + ".csv", 'w') as f:
        f.write(header)

        wr = csv.writer(f)
        [wr.writerow(e) for e in zip(*elements)]

def format_json(jn):
    """Format json contents."""
    jn = jn.replace(":", ": ")
    for arr in re.findall('\[\n\s+[A-z0-9$",\-\.\n\s]*\]', jn):
        jn = jn.replace(arr, re.sub(r',\n\s+', r',', arr))
    jn = re.sub(r'\[\n\s+([A-Za-z0-9,"$\.\-]+)\n\s+\]', r'[\1]', jn)
    jn = jn.replace("],[", "],\n            [")
    return jn

def format_time(time):
    """Format time as string."""
    return str(time).replace(".", "").zfill(3)

def is_tar(file):
    """Check if file has .tar.xz extension."""
    return file[-7:] == ".tar.xz"

def is_json(file):
    """Check if file has .json extension."""
    return file[-5:] == ".json"

def is_pkl(file):
    """Check if file has .pkl extension."""
    return file[-4:] == ".pkl"