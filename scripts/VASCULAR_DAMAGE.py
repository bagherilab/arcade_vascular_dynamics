from utilities import load
import tarfile

class VASCULAR_DAMAGE():
    NAME = "VASCULAR_DAMAGE"

    CONTEXTS = [
        ('C', '', [-1]),
        ('CH', 'X', [-1, 1])
    ]

    VALUES = ['000', '010', '020', '030', '040', '050', '060', '070', '080', '090', '100']

    FRACTIONS = ['000', '100']


    @staticmethod
    def run(output_path, input_path, func, name=NAME,
            contexts=CONTEXTS, fractions=FRACTIONS, values=VALUES, timepoints=[], seeds=[]):
        outfile = output_path + name + "/" + name

        for context, suffix, exclude in contexts:
            for fraction in fractions:
                for value in values:
                        infile = input_path + "_".join([name + "/" + name, context, "damage" + value, fraction]) + ".pkl"
                        code = "_" + "_".join([context + suffix, value, fraction])
                        print(name + " : " + code)

                        loaded = X.load(infile)
                        func(*loaded, outfile, code, exclude=exclude, timepoints=timepoints, seeds=seeds)

    @staticmethod
    def loop(output_path, func1, func2, metric, name=NAME,
             contexts=CONTEXTS, fractions=FRACTIONS, values=VALUES, timepoints=[]):
        outfile = output_path + name + "/" + name
        out = { "data": [] }

        for context, suffix, exclude in contexts:
            for t in timepoints:
                for fraction in fractions:
                    for value in values:
                        code = "_" + "_".join([context + suffix, value, fraction])
                        func1(outfile, out, { "time": t, "context": context + suffix, "value": value, "frac": fraction }, metric, code)

        func2(outfile + "_" + metric, out)

    @staticmethod
    def load(output_path, input_path, func, extension="", name=NAME,
             contexts=CONTEXTS, fractions=FRACTIONS, values=VALUES, timepoints=[], seeds=[]):
        outfile = output_path + name + "/" + name + extension

        for context, suffix, exclude in contexts:
            for fraction in fractions:
                for value in values:
                        infile = input_path + "_".join([name + extension + "/" + name, context, "damage" + value, fraction]) + extension + ".tar.xz"
                        code = "_" + "_".join([context, value, fraction])
                        print(name + " : " + code)

                        data = tar.open(infile)
                        func(data, timepoints, { "context": context, "value": value, "frac": fraction }, outfile, code)
