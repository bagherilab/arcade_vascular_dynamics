from utilities import load
import tarfile

class VASCULAR_FUNCTION():
    NAME = "VASCULAR_FUNCTION"

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
        outfile = output_path + name + "/" + name

        for context, suffix, exclude in contexts:
            for g, graph in graphs:
                code = "_" + "_".join([context + suffix, graph])
                infile = input_path + "_".join([name + "/" + name, context, g + graph]) + ".pkl"
                print(name + " : " + code)

                loaded = load(infile)
                func(*loaded, outfile, code, exclude=exclude, timepoints=timepoints, seeds=seeds)

    @staticmethod
    def loop(output_path, func1, func2, metric, name=NAME,
             contexts=CONTEXTS, graphs=GRAPHS, timepoints=[]):
        outfile = output_path + name + "/" + name
        out = { "data": [] }

        for context, suffix, exclude in contexts:
            for t in timepoints:
                for g, graph in graphs:
                    code = "_" + "_".join([context + suffix, graph])
                    func1(outfile, out, { "time": t, "context": context + suffix, "graphs": graph }, metric, code)

        func2(outfile + "_" + metric, out)

    @staticmethod
    def load(output_path, input_path, func, extension="", name=NAME,
             contexts=CONTEXTS, graphs=GRAPHS, timepoints=[], seeds=[]):
        outfile = output_path + name + "/" + name + extension

        for context, suffix, exclude in contexts:
            for g, graph in graphs:
                code = "_" + "_".join([context, graph])
                infile = input_path + "_".join([name + extension + "/" + name, context, g + graph]) + extension + ".tar.xz"
                print(name + " : " + code)

                data = tar.open(infile)
                func(data, timepoints, { "context": context, "graphs": graph }, outfile, code)
