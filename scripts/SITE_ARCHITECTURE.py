from utilities import load
import tarfile

class SITE_ARCHITECTURE():
    NAME = "SITE_ARCHITECTURE"

    CONTEXTS = [
        ('C', '', [-1]),
        ('CH', 'X', [-1, 1])
    ]

    SITES = ["SOURCE", "PATTERN", "GRAPH"]

    LAYOUTS = {
        "SOURCE": [
            "constant",
            "point0",
            "point2","point4","point6","point8","point10",
            "point12","point14","point16","point18","point20",
            "point22","point24","point26","point28","point30",
            "point32","point34","point36","point38",
            "grid2","grid3","grid4","grid5",
            "grid6","grid7","grid8","grid9","grid10",
            "grid11","grid12","grid13","grid14","grid15",
            "grid16","grid17","grid18","grid19","grid20",
            "grid30","grid40","grid50","grid100","grid200",
            "x1y2","x1y3","x1y4","x1y5",
            "x2y1","x2y3","x2y4","x2y5",
            "x3y1","x3y2","x3y4","x3y5",
            "x4y1","x4y2","x4y3","x4y5",
            "x5y1","x5y2","x5y3","x5y4",
            "nosite"
        ],
        "PATTERN": [""],
        "GRAPH": ["Lav", "Lava", "Lvav", "Sav", "Savav"]
    }

    @staticmethod
    def run(output_path, input_path, func, name=NAME,
            contexts=CONTEXTS, sites=SITES, layouts=LAYOUTS, timepoints=[], seeds=[]):
        outfile = output_path + name + "/" + name

        for context, suffix, exclude in contexts:
            for site in sites:
                for layout in layouts[site]:
                    if layout == "":
                        code = "_" + "_".join([context + suffix, site])
                        infile = input_path + "_".join([name + "/" + name, context, site]) + ".pkl"
                    else:
                        code = "_" + "_".join([context + suffix, site, layout])
                        infile = input_path + "_".join([name + "/" + name, context, site, layout]) + ".pkl"

                    print(name + " : " + code)

                    loaded = load(infile)
                    func(*loaded, outfile, code, exclude=exclude, timepoints=timepoints, seeds=seeds)

    @staticmethod
    def loop(output_path, func1, func2, metric, name=NAME,
             contexts=CONTEXTS, sites=SITES, layouts=LAYOUTS, timepoints=[]):
        outfile = output_path + name + "/" + name
        out = { "data": [] }
        
        for context, suffix, exclude in contexts:
            for t in timepoints:
                for site in sites:
                    for layout in layouts[site]:
                        if layout == "":
                            code = "_" + "_".join([context + suffix, site])
                            s = site
                        else:
                            code = "_" + "_".join([context + suffix, site, layout])
                            s = site + "_" + layout

                        func1(outfile, out, { "time": t, "context": context + suffix, "sites": s }, metric, code)

        func2(outfile + "_" + metric, out)

    @staticmethod
    def load(output_path, input_path, func, extension="", name=NAME,
             contexts=CONTEXTS, sites=SITES, layouts=LAYOUTS, timepoints=[], seeds=[]):
        outfile = output_path + name + "/" + name + extension

        for context, suffix, exclude in contexts:
            for site in sites:
                for layout in layouts[site]:
                    if layout == "":
                        code = "_" + "_".join([context, site])
                        infile = input_path + "_".join([name + extension + "/" + name, context, site]) + extension + ".tar.xz"
                        s = site
                    else:
                        code = "_" + "_".join([context, site, layout])
                        infile = input_path + "_".join([name + extension + "/" + name, context, site, layout]) + extension + ".tar.xz"
                        s = site + "_" + layout
                    
                    print(name + " : " + code)

                    data = tarfile.open(infile)
                    func(data, timepoints, { "context": context, "sites": s }, outfile, code)