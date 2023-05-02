import os

import pandas as pd

from util import *
from setup import *

engine, trans = CUTEST_UTIL.establish_connection()

AGG_RESULT = f"""
    (select method,
             max(`update`)                 as up_max,
             min(`update`)                 as up_min,
             rn                            as version,
             sum(status)                   as nf,
             avg(if(status = 0, 200, t)) as tf,
             avg(if(status = 0, 20000, k)) as kf,
             avg(kf)     as kff,
             avg(kg)     as kfg,
             avg(kh)     as kfh
      from ranked_messages
      {FILTER}
      group by method, rn)
         as t
         left join (select method,
                           rn                         as version,
                           exp(avg(ln(if(status = 0, 200, t) + 1))) - 1   as tg,
                           exp(avg(ln(if(status = 0, 20000, k) + 50))) - 50 as kg,
                           exp(avg(ln(kf + 50))) - 50 as kgf,
                           exp(avg(ln(kg + 50))) - 50 as kgg,
                           exp(avg(ln(kh + 50))) - 50 as kgh
                    from ranked_messages
                    {FILTER}
                    group by method, rn) as tt
                   on tt.method = t.method and tt.version = t.version
"""
RANKING = f"""WITH ranked_messages AS (
    SELECT m.*, ROW_NUMBER() OVER (PARTITION BY name,param,method ORDER BY `update` DESC) AS rn
    FROM cutest.result AS m where method not in {UNSELECT_METHOD})
"""
# from sql
# query last
sql_query_all = f"""
    {RANKING}
    select *
    from ranked_messages
    {FILTER}
    and rn = 1
    and method not in {UNSELECT_METHOD};
    """
df = pd.read_sql(
    sql_query_all,
    con=engine,
).set_index(["name", "n", "method"])

version_number = int(df["update"].max().timestamp() / 100)
fdir = f"./{version_number}"
if not os.path.exists(fdir):
    os.mkdir(fdir)
# stack view
dfa = df[INFO_CUTEST_RESULT.COLUMNS_PERF].unstack(level=-1)

# to latex tables
dfl = df[INFO_CUTEST_RESULT.COLUMNS_PERF].assign(
    # fix entry formats
    **{
        k: df[k].apply(v)
        for k, v in INFO_CUTEST_RESULT.COLUMNS_FULL_TABLE_LATEX_WT_FORMATTER.items()
    }
)
dfl_stacked = dfl.unstack(level=-1)
latex_full_table_str1 = INFO_CUTEST_RESULT.produce_latex_long_table(
    dfl_stacked,
    ["k", "t"],
    caption="Complete Results on CUTEst Dataset, iteration \& time",
    label="tab.cutest.kt",
    path=os.path.join(fdir, "complete.kt.tex"),
)
latex_full_table_str2 = INFO_CUTEST_RESULT.produce_latex_long_table(
    dfl_stacked,
    ["fx", "df"],
    caption="Complete Results on CUTEst Dataset, function value \& norm of the gradient",
    label="tab.cutest.fx",
    path=os.path.join(fdir, "complete.fg.tex"),
)

# aggregated stats
COMMENTS = r"""Performance of different algorithms on the CUTEst dataset. 
    Note $\overline t, \overline k$ are mean running time and iteration of successful instances;
    $\overline t_{G}, \overline k_{G}$ are scaled geometric means (scaled by 1 second and 50 iterations, respectively).
    If an instance is the failed, its iteration number and solving time are set to $20,000$. 
"""
df_geo_perf = pd.read_sql(
    f"""
{RANKING}
select t.method,
       t.nf,
       t.tf,
       t.kf,
       t.kff,
       t.kfg,
       t.kfh,
       tt.tg,
       tt.kg,
       tt.kgf,
       tt.kgg,
       tt.kgh
from {AGG_RESULT}
where tt.version=1;
    """,
    con=engine,
).set_index(["method"])

latex_geo_sum_str = df_geo_perf.rename(
    columns=INFO_CUTEST_RESULT.COLUMNS_RENAMING
).to_latex(
    longtable=True,
    escape=False,
    multirow=True,
    caption=COMMENTS,
    label="tab.perf.geocutest",
    buf=os.path.join(fdir, "perf.history.tex"),
    float_format="%.2f",
    sparsify=True,
)

df_geo_perf = pd.read_sql(
    f"""
{RANKING}
select t.method,
       t.nf,
       tt.tg,
       tt.kg,
       tt.kgf,
       tt.kgg,
       tt.kgh
from {AGG_RESULT}
where tt.version=1;
    """,
    con=engine,
).set_index(["method"])

latex_geo_sum_str = df_geo_perf.rename(
    columns=INFO_CUTEST_RESULT.COLUMNS_RENAMING
).to_latex(
    longtable=False,
    escape=False,
    multirow=True,
    caption=COMMENTS,
    label="tab.perf.geocutest",
    buf=os.path.join(fdir, "perf.geo.tex"),
    float_format="%.2f",
    sparsify=True,
)

df_geo_perf = pd.read_sql(
    f"""
{RANKING}
select t.method,
       t.nf,
       t.tf,
       t.kf,
       t.kff,
       t.kfg,
       t.kfh
from {AGG_RESULT}
where tt.version=1;
    """,
    con=engine,
).set_index(["method"])

latex_geo_sum_str = df_geo_perf.rename(
    columns=INFO_CUTEST_RESULT.COLUMNS_RENAMING
).to_latex(
    longtable=False,
    escape=False,
    multirow=True,
    caption=COMMENTS,
    label="tab.perf.geocutest",
    buf=os.path.join(fdir, "perf.alg.tex"),
    float_format="%.2f",
    sparsify=True,
)


df_geo_perf = pd.read_sql(
    f"""
{RANKING}
select t.method,
       t.nf,
       t.tf,
       t.kf,
       t.kff,
       t.kfg,
       t.kfh,
       tt.tg,
       tt.kg,
       tt.kgf,
       tt.kgg,
       tt.kgh,
       tt.version
from {AGG_RESULT}
where tt.version=1;
    """,
    con=engine,
).set_index("method")
latex_geo_sum_str = df_geo_perf.rename(
    columns=INFO_CUTEST_RESULT.COLUMNS_RENAMING
).to_latex(
    longtable=True,
    escape=False,
    caption=COMMENTS,
    buf=os.path.join(fdir, "perf.tex"),
    float_format="%.2f",
    sparsify=True,
)


print(
    INFO_CUTEST_RESULT.QUICK_VIEW,
    file=open(os.path.join(fdir, f"{version_number}_view.tex"), "w"),
)


# standard profiling
import plotly.graph_objects as go

layout = go.Layout(
    plot_bgcolor="rgba(255, 255, 255, 1)",
    xaxis=dict(
        title=f"α",
        color="black",
    ),
    yaxis=dict(title=f"profile of α", color="black", range=[0, 1.1]),
    xaxis_type="log",
    font=dict(family="Latin Modern Roman", size=15),
    legend=dict(
        bordercolor="black",
        borderwidth=0.8,
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
)


df_b = (
    df.reset_index()
    .rename(columns={"t": "tsf"})
    .assign(t=lambda df: df["tsf"].apply(lambda x: max(x, 1)))
    .groupby(["name", "n"])
    .agg({"k": min, "t": min})
    .rename(columns={"k": "kb", "t": "tb"})
)

df_rho = (
    df.query("status==1")
    .join(df_b)
    .assign(rho_k=lambda df: df["k"] / df["kb"], rho_t=lambda df: df["t"] / df["tb"])
)
scale = df_rho.reset_index().groupby("method").status.sum().max()
methods = df_rho.index.get_level_values(2).unique().to_list()
metrics = {"rho_k": "k", "rho_t": "t"}
for m in metrics:
    data = []
    for method in methods:
        sr = df_rho[m][:, :, method]
        dist = sr.reset_index().groupby(m).count().cumsum()["n"]
        dist = dist / scale
        line = go.Line(
            x=dist.index,
            y=dist.values,
            name=INFO_CUTEST_RESULT.METHODS_RENAMING_REV[method],
            line=dict(width=1.5),
        )
        print(m, method)
        data.append(line)

    fig = go.Figure(data=data, layout=layout)
    style_grid = dict(
        showline=True,
        linewidth=1.2,
        linecolor="black",
        showgrid=True,
        gridwidth=0.5,
        gridcolor="grey",
        griddash="dashdot",
    )
    fig.update_xaxes(style_grid)
    fig.update_yaxes(style_grid)
    fig.write_image(f"{version_number}/{version_number}-{m}.png", scale=3)
    fig.write_html(f"{version_number}/{version_number}-{m}.html")

##############################################################################################

print("*" * 50)
print(f"results dump to: \n {version_number}/{version_number}_view.tex")
print(
    f"compile using: \n latexmk -xelatex -cd  {version_number}/{version_number}_view.tex"
)

print("*" * 50)
