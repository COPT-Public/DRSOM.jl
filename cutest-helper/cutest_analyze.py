import os

import pandas as pd

from util import *


UNSELECT_METHOD = r"('\\drsomh', '\\drsom')"
# from csv
# df = pd.read_csv("./cutest-2022111217.csv")
# df = df.set_index(["name", "n", "method"])
# from sql
# query last
sql_query_all = f"""
    WITH ranked_messages AS (
    SELECT m.*, ROW_NUMBER() OVER (PARTITION BY name,param,method ORDER BY `update` DESC) AS rn
    FROM cutest.result AS m where method not in {UNSELECT_METHOD}
    )
    select *
    from ranked_messages
    where k is not null
    and df <= 1e-5 # status = 1
    and t <= 100
    and n <= 200
    and rn = 1
    and method not in {UNSELECT_METHOD};
    """
engine, trans = CUTEST_UTIL.establish_connection()
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
    """WITH ranked_messages AS (SELECT m.*, ROW_NUMBER() OVER (PARTITION BY name,param,method ORDER BY `update` DESC) AS rn
                         FROM cutest.result AS m)
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
from (select method,
             rn          as version,
             sum(status) as nf,
             avg(t)      as tf,
             avg(k)      as kf,
             avg(kf)      as kff,
             avg(kg)      as kfg,
             avg(kh)      as kfh
      from ranked_messages
      where k is not null
        and df <= 1e-5 # status = 1
        and t <= 100
        and n <= 200
      group by method, rn)
         as t
         left join (select method,
                           rn                        as version,
                           exp(avg(ln(t + 5))) - 5   as tg,
                           exp(avg(ln(k + 50))) - 50 as kg,
                           exp(avg(ln(kf + 50))) - 50 as kgf,
                           exp(avg(ln(kg + 50))) - 50 as kgg,
                           exp(avg(ln(kh + 50))) - 50 as kgh
                    from ranked_messages
                    where t <= 100
                      and n <= 200
                    group by method, rn) as tt
                   on tt.method = t.method and tt.version = t.version;
    """,
    con=engine,
).set_index(["method", "version"])

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
    f"""WITH ranked_messages AS (SELECT m.*, ROW_NUMBER() OVER (PARTITION BY name,param,method ORDER BY `update` DESC) AS rn
                         FROM cutest.result AS m)
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
from (select method,
             rn          as version,
             sum(status) as nf,
             avg(t)      as tf,
             avg(k)      as kf,
             avg(kf)      as kff,
             avg(kg)      as kfg,
             avg(kh)      as kfh
      from ranked_messages
      where k is not null
        and df <= 1e-5 # status = 1
        and t <= 100
        and n <= 200
        and rn = 1
        and method not in {UNSELECT_METHOD}
      group by method, rn)
         as t
         left join (select method,
                           rn                        as version,
                           exp(avg(ln(t + 5))) - 5   as tg,
                           exp(avg(ln(k + 50))) - 50 as kg,
                           exp(avg(ln(kf + 50))) - 50 as kgf,
                           exp(avg(ln(kg + 50))) - 50 as kgg,
                           exp(avg(ln(kh + 50))) - 50 as kgh
                    from ranked_messages
                    where t <= 100
                      and n <= 200
                      and rn = 1
                    group by method, rn) as tt
                   on tt.method = t.method and tt.version = t.version;
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
