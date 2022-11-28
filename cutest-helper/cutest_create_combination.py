# use this script to create all possible combinations of params
# a cutest problem may have multiple params
# use `generate_cutest.sh` to create the csv
# @note:
#  invocation:
# $ python *.py the_file_you_want_to_input.csv
#
from sqlalchemy import create_engine
import pandas as pd
import sys


# dump basic params one per line
engine = create_engine("mysql://root@127.0.0.1:3306", echo=True)
df = pd.read_csv(sys.argv[1], dtype=str)
df["key"] = df["comb"].apply(lambda x: x.split("=")[0])
df.to_sql("problem_params", con=engine, schema="cutest", if_exists="replace")

# create full combination.
ss = df.groupby(["name"])
vv = []
for k, v in ss:
    param_keys = v["key"].unique()
    sss = v.groupby(["key"])["comb"].apply(set)
    print(sss.values)
    import itertools

    for tp in itertools.product(*sss.values):
        vv.append([k, (",".join(tp))])

dfs = pd.DataFrame(data=vv, columns=["name", "full_comb"])
dfs.to_sql("problem_params_comb", con=engine, schema="cutest", if_exists="replace")
