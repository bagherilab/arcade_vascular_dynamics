from utilities import load
import tarfile

class ESTIMATED_HEMODYNAMICS():
    NAME = "ESTIMATED_HEMODYNAMICS"

    CONTEXTS = [
        ('C', '', [-1]),
        ('CH', 'X', [-1, 1])
    ]

    WEIGHTS = ["weight_gradient", "weight_local", "weight_flow"]

    SCALES = ['000', '010', '020', '030', '040', '050', '060', '070', '080', '090', '100',
        '110', '120', '130', '140', '150', '160', '170', '180', '190', '200']

    @staticmethod
    def run(output_path, input_path, func, name=NAME,
            contexts=CONTEXTS, weights=WEIGHTS, scales=SCALES, timepoints=[], seeds=[]):
        outfile = f"{output_path}{name}/{name}"

        for context, suffix, exclude in contexts:
            for weight in weights:
                for scale in scales:
                    code = f"_{context}{suffix}_{weight}_{scale}"
                    infile = f"{input_path}{name}/{name}_{context}_{weight}_{scale}.pkl"
                    print(f"{name} : {code}")

                    loaded = load(infile)
                    func(*loaded, outfile, code, exclude=exclude, timepoints=timepoints, seeds=seeds)

    @staticmethod
    def loop(output_path, func1, func2, metric, name=NAME,
             contexts=CONTEXTS, weights=WEIGHTS, scales=SCALES, timepoints=[]):
        outfile = f"{output_path}{name}/{name}"
        out = { "data": [] }

        for context, suffix, exclude in contexts:
            for t in timepoints:
                for weight in weights:
                    for scale in scales:
                        code = f"_{context}{suffix}_{weight}_{scale}"
                        func1(outfile, out, { "time": t, "context": context + suffix, "weight": weight, "scale": scale }, metric, code)

        func2(f"{outfile}_{metric}", out)

    @staticmethod
    def load(output_path, input_path, func, extension="", name=NAME,
             contexts=CONTEXTS, weights=WEIGHTS, scales=SCALES, timepoints=[], seeds=[]):
        outfile = f"{output_path}{name}/{name}"

        for context, _, exclude in contexts:
            for weight in weights:
                for scale in scales:
                    code = f"_{context}_{weight}_{scale}"
                    infile = f"{input_path}{name}{extension}/{name}_{context}_{weight}_{scale}{extension}.tar.xz"
                    print(f"{name} : {code}")

                    data = tarfile.open(infile)
                    func(data, timepoints, { "context": context, "weight": weight, "scale": scale }, outfile, code)
