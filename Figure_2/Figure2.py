from dataclasses import dataclass
from pyecharts import options as opts
from pyecharts.charts import Sankey
import pandas as pd


data = pd.read_csv("Figure2-data.csv")

links = []
for i, d in data.iterrows():
    if d["source"] == "None ":
        d["source"] = "Inactive "
    if d["target"] == "None":
        d["target"] = "Inactive"
    link = {
        "source": d["source"],
        "target": d["target"],
        "value": round(float(d["value"]), 3)
    }
    print(link)
    links.append(link)

nodes = [
    {"name": "Left ", 'itemStyle': {'color': "#0E538F"}},
    {"name": "Left\r\nleaning ", 'itemStyle': {'color': "#4495DB"}},
    {"name": "Centre ", 'itemStyle': {'color': "#CFDB00"}},
    {"name": "Right\r\nleaning ", 'itemStyle': {'color': "#DB4742"}},
    {"name": "Right ", 'itemStyle': {'color': "#8F100B"}},
    {"name": "Fake and\r\nextreme bias ", 'itemStyle': {'color': "#282828"}},
    # {"name": "Inactive ", 'itemStyle': {'color': "#D3D3D3"}},

    {"name": "Left", 'itemStyle': {'color': "#0E538F"}},
    {"name": "Left\r\nleaning", 'itemStyle': {'color': "#4495DB"}},
    {"name": "Centre", 'itemStyle': {'color': "#CFDB00"}},
    {"name": "Right\r\nleaning", 'itemStyle': {'color': "#DB4742"}},
    {"name": "Right", 'itemStyle': {'color': "#8F100B"}},
    {"name": "Fake and\r\nextreme bias", 'itemStyle': {'color': "#282828"}},
    # {"name": "Inactive", 'itemStyle': {'color': "#D3D3D3"}},
]

c = (
    Sankey(init_opts=opts.InitOpts(height="600px", width="1400px"))
    .add(
        "",
        nodes=nodes,
        links=links,
        pos_left="20%",
        pos_top="20%",
        # focus_node_adjacency="allEdges",
        orient="vertical",
        node_width=20,
        node_gap=50,
        linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
        label_opts=opts.LabelOpts(position="top", font_size=15, font_family="FreeSans"),
    )
    .set_global_opts(
        # title_opts=opts.TitleOpts(title="Sankey-Vertical"),
        tooltip_opts=opts.TooltipOpts(trigger="item", trigger_on="mousemove"),
    )
    .set_series_opts(
        layoutIterations=0
    )
    .render("Figure2.html")
)