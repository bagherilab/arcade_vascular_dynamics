from .utilities import load, load_tar
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
        outfile = f"{output_path}{name}/{name}"

        for context, suffix, exclude in contexts:
            for fraction in fractions:
                for value in values:
                    code = f"_{context}{suffix}_{value}_{fraction}"
                    infile = f"{input_path}{name}/{name}_{context}_damage{value}_{fraction}.pkl"
                    print(f"{name} : {code}")

                    loaded = load(infile)
                    func(*loaded, outfile, code, exclude=exclude, timepoints=timepoints, seeds=seeds)

    @staticmethod
    def loop(output_path, func1, func2, extension, name=NAME,
             contexts=CONTEXTS, fractions=FRACTIONS, values=VALUES, timepoints=[]):
        outfile = f"{output_path}{name}/{name}"
        out = { "data": [] }
        tar = load_tar(outfile, extension)

        for context, suffix, exclude in contexts:
            for t in timepoints:
                for fraction in fractions:
                    for value in values:
                        code = f"_{context}{suffix}_{value}_{fraction}"
                        func1(outfile, out, { "time": t, "context": context + suffix, "value": value, "frac": fraction }, extension, code, tar=tar)

        func2(outfile, extension, out)

    @staticmethod
    def load(output_path, input_path, func, extension="", name=NAME,
             contexts=CONTEXTS, fractions=FRACTIONS, values=VALUES, timepoints=[], seeds=[]):
        outfile = f"{output_path}{name}/{name}"

        for context, _, exclude in contexts:
            for fraction in fractions:
                for value in values:
                    code = f"_{context}_{value}_{fraction}"
                    infile = f"{input_path}{name}{extension}/{name}_{context}_damage{value}_{fraction}{extension}.tar.xz"
                    print(f"{name} : {code}")

                    tar = tarfile.open(infile)
                    func(tar, timepoints, { "context": context, "value": value, "frac": fraction }, outfile, code)
