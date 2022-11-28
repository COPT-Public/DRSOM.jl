# @author: C. ZHANG
# this script dumps the csv table produced by JULIA to mysql
import sys
import time

import pandas as pd
from sqlalchemy import create_engine

from util import *

# argparse
fpath = sys.argv[1]


# loading
df = pd.read_csv(fpath)
if INFO_CUTEST_RESULT.PRIMARY_KEY not in df.columns:
    df[INFO_CUTEST_RESULT.PRIMARY_KEY] = df.apply(
        lambda row: f"{row['name']}-{row['n']}-{row['method']}-{int(pd.to_datetime(df.iloc[0]['update']).timestamp())}",
        axis=1,
    )
df = df.set_index([INFO_CUTEST_RESULT.PRIMARY_KEY])

# change to latex macro name
df["method"] = df["method"].apply(INFO_CUTEST_RESULT.METHODS_RENAMING.get)


engine, trans = CUTEST_UTIL.establish_connection()
try:
    # delete those rows that we are going to "upsert"
    cmd = f"delete from `{INFO_CUTEST_RESULT.NAME_SCHEMA}`.`{INFO_CUTEST_RESULT.NAME_TABLE}` where {INFO_CUTEST_RESULT.PRIMARY_KEY} in {tuple(df.index.to_list())}"
    print(cmd)
    engine.execute(cmd)
    trans.transaction.commit()

    # insert changed rows
    df[INFO_CUTEST_RESULT.COLUMNS].to_sql(
        INFO_CUTEST_RESULT.NAME_TABLE,
        engine,
        schema=INFO_CUTEST_RESULT.NAME_SCHEMA,
        if_exists="append",
    )
except:
    trans.transaction.rollback()
    raise
