import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine


# utility funcs
def convert_to_int_else_slash(x):
    try:
        return int(np.floor(x)).__str__()
    except:
        return "-"


def scaled_gmean(arr, scale=10):
    """compute scaled geometric mean,
    the scale=10 mimics the Hans's benchmark
    """
    return stats.gmean(arr + scale) - scale


# constants
class CUTEST_UTIL:
    @staticmethod
    def establish_connection():
        engine = create_engine("mysql://root@127.0.0.1:3306", echo=True)
        trans = engine.begin()
        return engine, trans


class INFO_CUTEST(object):
    NAME_SCHEMA = "cutest"


class INFO_CUTEST_RESULT(INFO_CUTEST):
    PARAM_GEOMETRIC_SCALER = 1
    NAME_TABLE = "result"
    PRIMARY_KEY = "id"
    COLUMNS = [
        "precision",
        "name",
        "param",
        "n",
        "method",
        "k",
        "kf",
        "kg",
        "kh",
        "df",
        "fx",
        "t",
        "status",
    ]
    COLUMNS_PERF = ["k", "kf", "kg", "kh", "df", "fx", "t", "status"]
    COLUMNS_RENAMING = {
        "n": "$n$",
        "k": "$k$",
        "t": "$t$",
        "df": "$\|g\|$",
        "fx": "$f$",
        # agg
        "kf": "$\\overline k$",
        "kff": "$\\overline k^f$",
        "kfg": "$\\overline k^g$",
        "kfh": "$\\overline k^H$",
        "tf": "$\\overline t$",
        "nf": "$\\mathcal K$",
        "kg": "$\\overline k_G$",
        "kgf": "$\\overline k_G^f$",
        "kgg": "$\\overline k_G^g$",
        "kgh": "$\\overline k_G^H$",
        "tg": "$\\overline t_G$",
    }
    COLUMNS_FULL_TABLE_LATEX_WT_FORMATTER = {
        "k": convert_to_int_else_slash,
        "df": lambda x: f"{np.nan_to_num(x, np.inf):.1e}",
        "fx": lambda x: f"{np.nan_to_num(x, np.inf):.1e}",
        "t": lambda x: f"{np.nan_to_num(x, np.inf):.1e}",
    }
    METHODS_RENAMING = {
        "DRSOM": r"\drsom",
        "NewtonTR": r"\newtontr",
        "HSODM": r"\hsodm",
        "DRSOMHomo": r"\drsomh",
        "HSODMArC": r"\hsodmarc",
        "LBFGS": r"\lbfgs",
        "CG": r"\cg",
        "ARC": r"\arc",
        "GD": r"\gd",
        "TRACE": r"\itrace",  # avoid conflicts with amsmath
    }
    METHODS_RENAMING_REV = {
        "\\drsom": "DRSOM",
        "\\newtontr": "NewtonTR",
        "\\hsodm": "HSODM",
        "\\lbfgs": "LBFGS",
        "\\cg": "CG",
        "\\arc": "ARC",
        "\\gd": "GD",
    }

    @staticmethod
    def produce_latex_long_table(df: pd.DataFrame, keys, caption, label, path):
        return (
            df[keys]
            .rename(
                index=INFO_CUTEST_RESULT.COLUMNS_RENAMING,
                columns=INFO_CUTEST_RESULT.COLUMNS_RENAMING,
            )
            .swaplevel(0, 1, 1)
            .sort_index(1)
            .to_latex(
                longtable=True, escape=False, caption=caption, label=label, buf=path
            )
        )

    QUICK_VIEW = r"""
\documentclass{article}
\usepackage{lscape,longtable,multirow}
\usepackage{booktabs,caption}

\newcommand{\hsodm}{\textrm{HSODM}}
\newcommand{\hsodmarc}{\textrm{HSODMArC}}
\newcommand{\drsom}{\textrm{DRSOM}}
\newcommand{\drsomh}{\textrm{DRSOM-H}}
\newcommand{\lbfgs}{\textrm{LBFGS}}
\newcommand{\newtontr}{\textrm{Newton-TR}}
\newcommand{\cg}{\textrm{CG}}
\newcommand{\arc}{\textrm{ARC}}
\newcommand{\itrace}{\textrm{TRACE}}
\newcommand{\gd}{\textrm{GD}}

\begin{document}

\begin{landscape}
    \input{perf.alg}
    \input{perf.geo}
    \clearpage
    \input{perf}
    \input{perf.history}
    \scriptsize
    \input{complete.kt}
    \input{complete.fg}
\end{landscape}
\end{document}
    """
