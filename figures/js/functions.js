// PROCESSORS ==================================================================

function processMake(layout, selected, order, name) {
    var file = function(order, inds) {
        var f = (layout.length > 0 ? order.map(function(e) { return inds[e]; }) : []);
        return name(f);
    }

    switch(layout.length) {
        case 0:
            return function() {
                return { "x": 0, "y": 0, "i": [], "file": file([]) }
            };
        case 1:
            return function(A, iA) {
                return { "x": iA, "y": 0, "file": file(order, [A]) };
            };
        case 2:
            return function(A, B, iA, iB) {
                return { "x": iA, "y": iB, "file": file(order, [A, B]) };
            };
        case 3:
            return function(A, B, C, iA, iB, iC) {
                var cn = selected[layout[2]].length;
                return { "x": iA*cn + iC, "y": iB, "file": file(order, [A, B, C]) };
            };
        case 4:
            return function(A, B, C, D, iA, iB, iC, iD) {
                var cn = selected[layout[2]].length;
                var dn = selected[layout[3]].length;
                return { "x": iA*cn + iC, "y": iB*dn + iD, "file": file(order, [A, B, C, D]) };
            };
        case 5:
            return function(A, B, C, D, E, iA, iB, iC, iD, iE) {
                var cn = selected[layout[2]].length;
                var dn = selected[layout[3]].length;
                var en = selected[layout[4]].length;
                return { "x": iA*cn*en + iC*en + iE, "y": iB*dn + iD, "file": file(order, [A, B, C, D, E]) };
            };
        case 6:
            return function(A, B, C, D, E, F, iA, iB, iC, iD, iE, iF) {
                var cn = selected[layout[2]].length;
                var dn = selected[layout[3]].length;
                var en = selected[layout[4]].length;
                var fn = selected[layout[5]].length;
                return { "x": iA*cn*en + iC*en + iE, "y": iB*dn*fn + iD*fn + iF, "file": file(order, [A, B, C, D, E, F]) };
            };
    }
}

function processGrid(layout, selected, make) {
    switch(layout.length) {
        case 0:
            return {
                "files": files = [make()],
                "marginLeft": 5,
                "marginTop": 5,
                "nCols": 1,
                "nRows": 1
            }
        case 1:
            return {
                "files": compileSingleFiles(layout, selected, make),
                "marginLeft": 5,
                "marginTop": LABEL_SIZE,
                "nCols": selected[layout[0]].length,
                "nRows": 1
            }
        case 2:
            return {
                "files": compileDoubleFiles(layout, selected, make),
                "marginLeft": LABEL_SIZE,
                "marginTop": LABEL_SIZE,
                "nCols": selected[layout[0]].length,
                "nRows": selected[layout[1]].length
            }
        case 3:
            return {
                "files": compileTripleFiles(layout, selected, make),
                "marginLeft": LABEL_SIZE,
                "marginTop": 2*LABEL_SIZE + LABEL_PADDING,
                "nCols": selected[layout[0]].length*selected[layout[2]].length,
                "nRows": selected[layout[1]].length
            }
        case 4:
            return {
                "files": compileQuadrupleFiles(layout, selected, make),
                "marginLeft": 2*LABEL_SIZE + LABEL_PADDING,
                "marginTop": 2*LABEL_SIZE + LABEL_PADDING,
                "nCols": selected[layout[0]].length*selected[layout[2]].length,
                "nRows": selected[layout[1]].length*selected[layout[3]].length
            }
        case 5:
            return {
                "files": compileQuintupleFiles(layout, selected, make),
                "marginLeft": 2*LABEL_SIZE + LABEL_PADDING,
                "marginTop": 3*LABEL_SIZE + 2*LABEL_PADDING,
                "nCols": selected[layout[0]].length*selected[layout[2]].length*selected[layout[4]].length,
                "nRows": selected[layout[1]].length*selected[layout[3]].length
            }
        case 6:
            return {
                "files": compileSextupleFiles(layout, selected, make),
                "marginLeft": 3*LABEL_SIZE + 2*LABEL_PADDING,
                "marginTop": 3*LABEL_SIZE + 2*LABEL_PADDING,
                "nCols": selected[layout[0]].length*selected[layout[2]].length*selected[layout[4]].length,
                "nRows": selected[layout[1]].length*selected[layout[3]].length*selected[layout[5]].length
            }
    }
}

// PLOTTERS ====================================================================

function plotPath(g, S) {
    g.append("path")
        .attr("d", function(d) {
            var xscale = (d.scale ? S.xscale[d.scale.x] : S.xscale);
            var yscale = (d.scale ? S.yscale[d.scale.y] : S.yscale);
            var makePath = d3.line()
                .x(function(m) { return xscale(m); })
                .y(function(m,i) { return yscale(d.y[i]); })
            return makePath(d.x);
        })
        .attr("fill", "none")
        .attr("stroke", function(d) { return (d.stroke ? d.stroke : "#555"); })
        .attr("stroke-width", function(d) { return (d.width ? d.width : 1); })
        .attr("opacity", function(d) { return (d.opacity ? d.opacity : null); })
        .attr("stroke-linecap", function(d) { return (d.linecap ? d.linecap : null); })
        .attr("stroke-dasharray", function(d) { return (d.dash ? d.dash : null); })
        .attr("stroke-dashoffset", function(d) { return (d.offset ? d.offset : null); });
}

function plotArea(g, S) {
    g.append("path")
        .attr("d", function(d) {
            var xscale = (d.scale ? S.xscale[d.scale.x] : S.xscale);
            var yscale = (d.scale ? S.yscale[d.scale.y] : S.yscale);
            return d3.area()
                .x(function(m) { return xscale(m); })
                .y0(function(m, i) { return yscale(d.max[i]); })
                .y1(function(m, i) { return yscale(d.min[i]); })(d.x);
        })
        .attr("fill", function(d) { return (d.fill ? d.fill : "#555"); })
        .attr("stroke", function(d) { return (d.stroke ? d.stroke : "none"); })
        .attr("stroke-width", function(d) { return (d.width ? d.width : "none"); })
        .attr("opacity", function(d) { return (d.opacity ? d.opacity : null); })
}

function plotViolin(g, S) {
    var lineForward = d3.line()
        .x(d => S.xscale(d.fx + d.dx))
        .y(d => S.yscale(d.fy + d.dy))
        .curve(d3.curveCardinal.tension(0.7));
    var lineBackward = d3.line()
        .x(d => S.xscale(d.rx + d.dx))
        .y(d => S.yscale(d.ry + d.dy))
        .curve(d3.curveCardinal.tension(0.7));

    var makeViolin = function(d) {
        if (d.direction == "horizontal") {
            var hist = d.x.map(function(e, i) { return {
                "v": e,
                "fy": e, "fx": d.y[i],
                "ry": -e, "rx": d.y[i],
                "dy": d.offset + 0.5, "dx": 0
            }})
        } else {
            var hist = d.y.map(function(e, i) { return {
                "v": e,
                "fx": e, "fy": d.x[i],
                "rx": -e, "ry": d.x[i],
                "dx": d.offset + 0.5, "dy": 0
            }})
        }

        var splits = [];
        var ind = 0;

        for (var i = 1; i < hist.length - 1; i++) {
            if (hist[i - 1].v == 0 && hist[i].v != 0) { ind = i; }
            if (hist[i + 1].v == 0 && hist[i].v != 0) {
                var sub = hist.slice(ind - 1, i + 2);
                splits.push(lineForward(sub) +  lineBackward(sub.reverse()).replace("M", "L") + "z")
            }
        }

        return splits.join(" ");
    };

    g.append("path")
        .attr("d", function(d) { return makeViolin(d); })
        .attr("fill", function(d) { return (d.fill ? d.fill : "none"); })
        .attr("stroke", function(d) { return (d.stroke ? d.stroke : "#555"); })
        .attr("stroke-width", function(d) { return (d.width ? d.width : 1); })
        .attr("stroke-dasharray", function(d) { return (d.dash ? d.dash : null); })
        .attr("opacity", function(d) { return (d.opacity ? d.opacity : null); });
}

function plotSymbol(g, S) {
    g.selectAll("use")
        .data(function(d) {
            if (d.scale) {
                var xscale = typeof S.xscale === "object" ? S.xscale[d.scale.x] : S.xscale;
                var yscale = typeof S.yscale === "object" ? S.yscale[d.scale.y] : S.yscale;

                return d.cx.map(function(e, i) {
                    return {
                        "link": d.link[i],
                        "cx": xscale(e),
                        "cy": yscale(d.cy[i]),
                        "fill": d.fill[i],
                        "stroke": (d.stroke ? d.stroke[i] : "none"),
                        "width": (d.width ? d.width[i] : "1px")
                    }
                })
            } else {
                var diam = S.axis.x.bounds[1]*2;
                var scale = Math.min(S.subpanel.h/(diam + 1)/2, S.subpanel.w/(diam + 1)/2);
                return d.cx.map(function(e, i) {
                    return {
                        "link": d.link[i],
                        "cx": (S.subpanel.w/2 + scale*e),
                        "cy": (S.subpanel.h/2 + scale*d.cy[i]),
                        "fill": d.fill[i],
                        "stroke": d.stroke[i],
                        "width": (d.width ? d.width[i] : "1px")
                    }
                })
            }
        })
        .enter().append("use")
        .attr("transform", function(d) { return "translate(" + d.cx + "," + d.cy + ")"; })
        .attr("xlink:href", function(d) { return d.link; })
        .attr("fill", function(d) { return d.fill; })
        .attr("stroke", function(d) { return d.stroke; })
        .attr("stroke-width", function(d) { return d.width; })
}

function plotCircle(g, S) {
    var R = Math.min(5, Math.max(2, Math.min(S.subpanel.dw, S.subpanel.dh)/100));
    g.selectAll("circle")
        .data(function(d) {
            var xscale = (d.scale ? S.xscale[d.scale.x] : S.xscale);
            var yscale = (d.scale ? S.yscale[d.scale.y] : S.yscale);

            return d.x.map(function(e, i) {
                return {
                    "x": xscale(e),
                    "y": yscale(d.y[i]),
                    "r": (d.r ? (Array.isArray(d.r) ? d.r[i] : d.r ) : d.R ? Math.min(xscale(d.R), yscale(d.R)) : R),
                    "fill": (d.fill ? (Array.isArray(d.fill) ? d.fill[i] : d.fill) : "#555"),
                    "stroke": (d.stroke ? (Array.isArray(d.stroke) ? d.stroke[i] : d.stroke) : null),
                    "opacity": (d.opacity ? (Array.isArray(d.opacity) ? d.opacity[i] : d.opacity) : null),
                }
            })
        })
        .enter().append("circle")
            .attr("cx", function(d) { return d.x; })
            .attr("cy", function(d) { return d.y; })
            .attr("r", function(d) { return d.r; })
            .attr("fill", function(d) { return d.fill; })
            .attr("stroke", function(d) { return d.stroke; })
            .attr("opacity", function(d) { return d.opacity; })
}

function plotRect(g, S) {
    g.selectAll("rect")
        .data(function(d) {
            var xscale = (d.scale ? S.xscale[d.scale.x] : S.xscale);
            var yscale = (d.scale ? S.yscale[d.scale.y] : S.yscale);

            return d.x.map(function(e, i) {
                return {
                    "x": xscale(e) + (d.dx ? d.dx[i] : 0),
                    "y": yscale(d.y[i]) + (d.dy ? d.dy[i] : 0),
                    "w": xscale(d.width[i]) - xscale(0) + (d.dw ? d.dw[i] : 0),
                    "h": yscale(0) - yscale(d.height[i]) + (d.dh ? d.dh[i] : 0),
                    "fill": (d.fill ? (Array.isArray(d.fill) ? d.fill[i] : d.fill) : "#555"),
                    "stroke": (d.stroke ? (Array.isArray(d.stroke) ? d.stroke[i] : d.stroke) : null),
                    "opacity": (d.opacity ? (Array.isArray(d.opacity) ? d.opacity[i] : d.opacity) : null),
                    "dash": (d.dash ? (Array.isArray(d.dash) ? d.dash[i] : d.dash) : null),
                }
            })
        })
        .enter().append("rect")
            .attr("x", function(d) { return d.x; })
            .attr("y", function(d) { return d.y; })
            .attr("width", function(d) { return d.w; })
            .attr("height", function(d) { return d.h; })
            .attr("fill", function(d) { return d.fill; })
            .attr("stroke", function(d) { return d.stroke; })
            .attr("opacity", function(d) { return d.opacity; })
            .attr("stroke-dasharray", function(d) { return d.dash; })
}

// LABELERS ====================================================================

function labelGrid(S, P) {
    switch(S.layout.length) {
        case 0: return labelNone(S, P)
        case 1: return labelOne(S, P)
        case 2: return labelTwo(S, P)
        case 3: return labelThree(S, P)
    }
}

function labelNone(S, P) { return [] }

function labelOne(S, P) {
    var labels = [];
    var layout = S.layout;

    var L = layout.map(function(e) {
        return S.selected[e].filter(function(f) { return f != ""; }); });

    var outerX = function(e, i) {
        return makeHorzLabel(S.panel.w, PANEL_PADDING/2 + S.panel.dw*i, 0,
            LABELS[layout[0]][e], shadeColor("#aaaaaa", i/L[0].length));
    }

    L[0].map(function(e, i) { labels.push(outerX(e, i)); });

    return labels;
}

function labelTwo(S, P) {
    var labels = [];
    var layout = S.layout;
    if (Array.isArray(S.layout[0])) { layout = S.selected.ordering; }

    var L = layout.map(function(e) {
        return S.selected[e].filter(function(f) { return f != ""; }); });

    var outerX = function(e, i) {
        return makeHorzLabel(S.panel.w, PANEL_PADDING/2 + S.panel.dw*i, 0,
            LABELS[layout[0]][e], shadeColor("#aaaaaa", i/L[0].length));
    }

    var outerY = function(e, i) {
        return makeVertLabel(S.panel.h, 0, PANEL_PADDING/2 + S.panel.dh*i,
            LABELS[layout[1]][e], shadeColor("#aaaaaa", i/L[1].length));
    }

    L[0].map(function(e, i) { labels.push(outerX(e, i)); });
    L[1].map(function(e, i) { labels.push(outerY(e, i)); });

    return labels;
}

function labelThree(S, P) {
    var labels = [];
    var layout = S.layout;
    if (Array.isArray(S.layout[0])) { layout = S.selected.ordering; }

    var L = layout.map(function(e) {
        return S.selected[e].filter(function(f) { return f != ""; }); });

    var outerX = function(e, i) {
        var W = S.panel.dw*L[2].length;
        return makeHorzLabel(W - PANEL_PADDING, PANEL_PADDING/2 + W*i, -LABEL_SIZE - LABEL_PADDING,
            LABELS[layout[0]][e], shadeColor("#aaaaaa", i/L[0].length));
    }

    var outerY = function(e, i) {
        return makeVertLabel(S.panel.h, 0, PANEL_PADDING/2 + S.panel.dh*i,
            LABELS[layout[1]][e], shadeColor("#aaaaaa", i/L[1].length));
    }

    L[0].map(function(e, i) { labels.push(outerX(e, i)); });
    L[1].map(function(e, i) { labels.push(outerY(e, i)); });
    if (L[2].length > 0) { makeInnerXLabels(S, P, L, labels, 2, layout); }

    return labels;
}

// DECORATORS ==================================================================

function decorateTicks(g, S, i, p) {
    addBorder(g, S.subpanel.w, S.subpanel.h, "#ccc");

    // Create and align groups for ticks.
    var dx = alignHorzAxis(S, i);
    var dy = alignVertAxis(S, i);

    // Create group to hold ticks.
    var G = S.G.append("g")
        .attr("id", "ticks")
        .attr("transform", "translate(" + dx + "," + dy + ")")

    var A = S.axis;

    // Create ticks.
    var ticks = [];
    ticks.push(makeHorzTicks(S, 0, S.subpanel.h, A.x));
    ticks.push(makeVertTicks(S, 0, 0, A.y));

    // Create axis labels.
    var labels = [];
    labels.push(makeHorzLabel(S.subpanel.w, 0, alignHorzText(S), A.x.title, "none"));
    labels.push(makeVertLabel(S.subpanel.h, alignVertText(S), 0, A.y.title, "none"));

    addTicks(G, ticks);
    addLabels(G, labels);
}
