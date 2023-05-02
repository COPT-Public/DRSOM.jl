# change this file to do profiling

# un-picked methods in the following
# e.g. second-order method only...
UNSELECT_METHOD = r"('\\lbfgs', '\\drsomh', '\\hsodmarc', '\\gd', '\\cg', '\\drsom')"

# filter the results satisfying the following condition...
OPTION = 1
if OPTION:
    FILTER = """
    where k <= 5000000
        and n <= 200
        and `precision` = 1e-5
        """
else:
    FILTER = """
        where k <= 5000000
            and `precision` = 1e-5
    """
