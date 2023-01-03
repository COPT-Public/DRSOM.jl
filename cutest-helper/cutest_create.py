from sqlalchemy import create_engine
import pycutest
import pandas as pd

engine = create_engine("mysql://127.0.0.1:3306", echo=True)

recs = [{"name": i, **pycutest.problem_properties(i)} for i in pycutest.find_problems()]
df = pd.DataFrame.from_records(recs)
df.to_sql("problem", con=engine, schema="cutest")
