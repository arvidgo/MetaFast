from bokeh.models import ColumnDataSource, LinearAxis, Whisker, Range1d, Select, MultiChoice
from bokeh.plotting import figure, output_file, show
from bokeh.transform import dodge, linear_cmap
from bokeh.layouts import column, layout
from bokeh.io import curdoc
from bokeh.palettes import *
from sys import argv
from math import floor, log
import pandas as pd

#### START TIMEPLOT

def generate_timeplot(timefiles):
       if not timefiles:
              return None
       cmds = []
       wclock = []
       mem = []
       cmddict = {}
       for fname in timefiles:
              with open(fname) as f:
                     lines = f.read().splitlines()
                     cmdstr = lines[0].split(": \"")[1].split()[0]
                     cmddict[cmdstr] = cmddict.get(cmdstr, 0) + 1
                     if(cmddict[cmdstr] != 1): cmdstr += str(cmddict[cmdstr])
                     cmds += [cmdstr]
                     wclock += [round(sum(x * float(t) for x, t in zip([60, 1], lines[4].split(": ")[1].split(":"))))]
                     mem += [int(lines[9].split(": ")[1])]

       #print(cmds, wclock, mem)
       maxmemlog = floor(log(max(mem), 1024))
       memstrs = ["k","m","g","t"]
       mem = [float(m)/(1024**maxmemlog) for m in mem]

       output_file("plots_res.html")
       graph = figure(title = "GNU time plots")

       data = {'command' : cmds,
              'time'   : wclock,
              'peak_rss': mem
       }
       source = ColumnDataSource(data=data)

       gnutime = figure(x_range=cmds, title="GNU time plots", y_range=(0, 1.05*max(wclock)), height=350)

       gnutime.vbar(x=dodge('command', -0.05, range=gnutime.x_range), top='time', source=source,
              width=0.1, color="#c9d9d3", legend_label="Time")
       gnutime.yaxis.axis_label = "Wall clock time (s)"

       gnutime.vbar(x=dodge('command',  0.05,  range=gnutime.x_range), top='peak_rss', source=source,
              width=0.1, color="#718dbf", legend_label="Peak RSS", y_range_name="rss")


       gnutime.extra_y_ranges['rss'] = Range1d(gnutime.y_range.start, 1.05*max(mem))
       mem_ax = LinearAxis(y_range_name="rss", axis_label=f"Peak RSS ({memstrs[maxmemlog]}bytes)")

       gnutime.x_range.range_padding = 0.1
       gnutime.xgrid.grid_line_color = None
       gnutime.add_layout(mem_ax, 'right')
       gnutime.add_layout(gnutime.legend[0], 'right')

       return gnutime

#### END TIMEPLOT

df = pd.read_csv(argv[1], sep="\t")
df = df[df["tool"] != "Gold standard"]

#### START WHISKERPLOT (IMAGE 1 ON RINGCENTRAL)
mask = lambda df, r, m: (df["rank"] == r) & (df["metric"] == m)# & (df["tool"] != "Gold standard")
get_results = lambda df, rank, metric: df[mask(df, rank, metric)][["tool", "value"]].groupby("tool")["value"]

markers = ["circle", "square", "diamond", "hex", "inverted_triangle", "triangle"] #bokeh.core.enums.MarkerType
colors_whisker = MediumContrast[5]
def filltolen(maxlen, lst):
       q, r = divmod(maxlen, len(lst))
       return lst*q + lst[:r]

def generate_whiskerplot(df, title, rank, xmetric, ymetric):
       xdata = get_results(df, rank, xmetric)
       ydata = get_results(df, rank, ymetric)
       numtools = len(xdata.unique())
       gsource = ColumnDataSource(data=dict(
              x=xdata.mean(), xupper = xdata.quantile(0.8), xlower = xdata.quantile(0.2),
              y=ydata.mean(), yupper = ydata.quantile(0.8), ylower = ydata.quantile(0.2),
              m=filltolen(numtools, markers), c=filltolen(numtools, colors_whisker), tool=list(xdata.mean().keys())
              ))
       gresult = figure(title=f"{title} - {rank}" if title else "", width=800, height=400)


       if(xdata.quantile(0.8).mean() != xdata.quantile(0.2).mean()): #if these means match there is no variance at all (ignoring extremely unlikely edgecase)
              xerror = Whisker(base="y", upper="xupper", lower="xlower", source=gsource,
                            level="annotation", line_width=2, dimension="width", line_color="c")
              xerror.upper_head.line_color = "c"
              xerror.lower_head.line_color = "c"
              gresult.add_layout(xerror)
       if(ydata.quantile(0.8).mean() != ydata.quantile(0.2).mean()): #if these means match there is no variance at all (ignoring extremely unlikely edgecase)
              yerror = Whisker(base="x", upper="yupper", lower="ylower", source=gsource,
                            level="annotation", line_width=2, dimension="height", line_color="c")
              yerror.upper_head.line_color = "c"
              yerror.lower_head.line_color = "c"
              gresult.add_layout(yerror)

       gresult.toolbar.autohide = True
       gresult.scatter(x="x", y="y", color="c", marker="m", size=10, legend_field="tool", source=gsource)
       gresult.add_layout(gresult.legend[0], 'right')
       gresult.xaxis.axis_label = xmetric
       gresult.yaxis.axis_label = ymetric
       return gresult

#### END WHISKERPLOT
#### START HEATMAP (IMAGE 2, LEFT SIDE ON RINGCENTRAL)

metrics_init = ["False negatives", "False positives", "Completeness", "Purity", "F1 score", "L1 norm error"]
lower_is_better = ['Bray-Curtis distance', 'False negatives', 'False positives', 'L1 norm error', 'Unweighted UniFrac error', 'Weighted UniFrac error']
#max_is_1 = ["Completeness", "Purity", "F1 score"]
tools_init = sorted(df.tool.unique())
fill_hm = Greens[7]
text_hm = grey(2)[::-1]

global_coloring = False
#normalize_1_coloring = True

def generate_heatmap(df, title, rank, tools=tools_init, metrics=metrics_init):
       hmsource = df[(df["rank"] == rank) & df.metric.isin(metrics) & df.tool.isin(tools)][["tool", "metric", "value"]]\
                                   .groupby(["tool", "metric"]).mean().round(2).reset_index()

       tools = sorted(tools, key=lambda t: -float(hmsource[(hmsource["metric"] == "F1 score") & (hmsource["tool"] == t)].value))

       p = figure(title=f"{title} - {rank}" if title else "", x_range=metrics, y_range=tools, width=900, height=400)

       p.grid.grid_line_color = None
       p.axis.axis_line_color = None
       p.toolbar.autohide = True

       for i,m in enumerate(metrics):
              #cmin, cmax = (0,1) if normalize_1_coloring or m in max_is_1 else \
              #      (hmsource[hmsource["metric"] == m].value.min(), hmsource[hmsource["metric"] == m].value.max())

              fillcol = fill_hm if global_coloring or m in lower_is_better else fill_hm[::-1]
              p.rect(x=i+.5, y="tool", width=1, height=1, source=hmsource[hmsource["metric"] == m],
                     fill_color=linear_cmap("value", fillcol, low=hmsource[hmsource["metric"] == m].value.min(),
                            high=hmsource[hmsource["metric"] == m].value.max()),
                     line_color=None)
              
              textcol = text_hm if global_coloring or m in lower_is_better else text_hm[::-1]
              p.text(x=i+.5, y="tool", text="value", source=hmsource[hmsource["metric"] == m], text_align="center",\
                     text_baseline="middle", text_color=linear_cmap("value", textcol, low=hmsource[hmsource["metric"] == m].value.min(),
                            high=hmsource[hmsource["metric"] == m].value.max()))
       return p

#### END HEATMAP
#### START INTERACTIVITY & LAYOUT

rwselect = Select(title="Whiskerplot rank:", value="species", options=sorted(df["rank"].unique()))
rhselect = Select(title="Heatmap rank:", value="species", options=sorted(df["rank"].unique()))

xselect = Select(title="X-Axis:", value="L1 norm error", options=sorted(df.metric.unique()))
yselect = Select(title="Y-Axis:", value="F1 score", options=sorted(df.metric.unique()))
mselect = MultiChoice(value=metrics_init, options=sorted(df.metric.unique()))
tselect = MultiChoice(value=tools_init, options=sorted(df.tool.unique()))

wtitle = ""; #"Whiskerplot"; //commented out for screenshot purposes
htitle = ""; #"Tool Heatmap"; //commented out for screenshot purposes

def update_whiskerplot(attr, old, new):
       layout.children[0].children[1] = generate_whiskerplot(df, wtitle, rwselect.value, xselect.value, yselect.value)

def update_heatmap(attr, old, new):
       mval = mselect.value if mselect.value else metrics_init
       tval = tselect.value if tselect.value else tools_init
       layout.children[2].children[0] = generate_heatmap(df, htitle, rhselect.value, tval, mval)

for i in [rwselect, xselect, yselect]:
       i.on_change("value", update_whiskerplot)
for i in [rhselect, mselect, tselect]:
       i.on_change("value", update_heatmap)

winputs = column(rwselect, xselect, yselect)
hinputs = column(rhselect, mselect, tselect)

layout = layout([winputs, generate_whiskerplot(df, wtitle, rwselect.value, xselect.value, yselect.value)],
       [hinputs],
       [generate_heatmap(df, htitle, rhselect.value), generate_timeplot(argv[2:])] if argv[2:]
              else [generate_heatmap(df, htitle, rhselect.value)])

curdoc().add_root(layout)
curdoc().title = "Metagenomic Benchmarking Results"

#### END INTERACTIVITY & LAYOUT
