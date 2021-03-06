from .utilities import load, load_tar
import tarfile

class EXACT_HEMODYNAMICS():
    NAME = "EXACT_HEMODYNAMICS"

    CONTEXTS = [
        ('C', '', [-1]),
        ('CH', 'X', [-1, 1])
    ]

    GRAPHS = [
        ("","PATTERN"),
        ("GRAPH_","Lav"),
        ("GRAPH_","Lava"),
        ("GRAPH_","Lvav"),
        ("GRAPH_","Sav"),
        ("GRAPH_","Savav")
    ]

    @staticmethod
    def run(output_path, input_path, func, name=NAME,
            contexts=CONTEXTS, graphs=GRAPHS, timepoints=[], seeds=[]):
        outfile = f"{output_path}{name}/{name}"

        for context, suffix, exclude in contexts:
            for g, graph in graphs:
                code = f"_{context}{suffix}_{graph}"
                infile = f"{input_path}{name}/{name}_{context}_{g}{graph}.pkl"
                print(f"{name} : {code}")

                loaded = load(infile)
                func(*loaded, outfile, code, exclude=exclude, timepoints=timepoints, seeds=seeds)

    @staticmethod
    def loop(output_path, func1, func2, extension, name=NAME,
             contexts=CONTEXTS, graphs=GRAPHS, timepoints=[]):
        outfile = f"{output_path}{name}/{name}"
        out = { "data": [] }
        tar = load_tar(outfile, extension)

        for context, suffix, exclude in contexts:
            for t in timepoints:
                for g, graph in graphs:
                    code = f"_{context}{suffix}_{graph}"
                    func1(outfile, out, { "time": t, "context": context + suffix, "graphs": graph }, extension, code, tar=tar)

        func2(outfile, extension, out)

    @staticmethod
    def load(output_path, input_path, func, extension="", name=NAME,
             contexts=CONTEXTS, graphs=GRAPHS, timepoints=[], seeds=[]):
        outfile = f"{output_path}{name}/{name}"

        for context, _, exclude in contexts:
            for g, graph in graphs:
                code = f"_{context}_{graph}"
                infile = f"{input_path}{name}{extension}/{name}_{context}_{g}{graph}{extension}.tar.xz"
                print(f"{name} : {code}")

                tar = tarfile.open(infile)
                func(tar, timepoints, { "context": context, "graphs": graph }, outfile, code)
