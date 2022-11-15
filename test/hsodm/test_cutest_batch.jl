###############
# project=> RSOM
# created Date=> Tu Mar 2022
# author=> <<author>
# -----
# last Modified=> Mon Apr 18 2022
# modified By=> Chuwen Zhang
# -----
# (c) 2022 Chuwen Zhang
# -----
# A script to test RSOM on smoothed L2-Lp minimization problems,
# Comparison of RSOM and A "real" second-order mothod (Newton-trust-region)
# For L2-Lp minimization, see the paper by X. Chen
# 1. Chen, X.=> Smoothing methods for nonsmooth, nonconvex minimization. Math. Program. 134, 71–99 (2012). https=>//doi.org/10.1007/s10107-012-0569-0
# 2. Chen, X., Ge, D., Wang, Z., Ye, Y.=> Complexity of unconstrained $$L_2-L_p$$ minimization. Math. Program. 143, 371–383 (2014). https=>//doi.org/10.1007/s10107-012-0613-0
# 3. Ge, D., Jiang, X., Ye, Y.=> A note on the complexity of Lp minimization. Mathematical Programming. 129, 285–299 (2011). https=>//doi.org/10.1007/s10107-011-0470-2
###############
include("./tools.jl")


# unconstrained problems
PROBLEMS = Dict(
    "ARGLINA" => [
        "M=200,N=200",
        "M=200,N=100",
        "M=200,N=10",
        "M=200,N=50",
        "M=100,N=200",
        "M=100,N=100",
        "M=100,N=10",
        "M=100,N=50",
        "M=20,N=200",
        "M=20,N=100",
        "M=20,N=10",
        "M=20,N=50",
        "M=400,N=200",
        "M=400,N=100",
        "M=400,N=10",
        "M=400,N=50"
    ], "ARGLINB" => [
        "M=200,N=200",
        "M=200,N=100",
        "M=200,N=10",
        "M=200,N=50",
        "M=100,N=200",
        "M=100,N=100",
        "M=100,N=10",
        "M=100,N=50",
        "M=20,N=200",
        "M=20,N=100",
        "M=20,N=10",
        "M=20,N=50",
        "M=400,N=200",
        "M=400,N=100",
        "M=400,N=10",
        "M=400,N=50"
    ],
    "ARGLINC" => [
        "M=200,N=200",
        "M=200,N=100",
        "M=200,N=10",
        "M=200,N=50",
        "M=100,N=200",
        "M=100,N=100",
        "M=100,N=10",
        "M=100,N=50",
        "M=20,N=200",
        "M=20,N=100",
        "M=20,N=10",
        "M=20,N=50",
        "M=400,N=200",
        "M=400,N=100",
        "M=400,N=10",
        "M=400,N=50"
    ],
    "ARGTRIGLS" => [
        "N=200",
        "N=100",
        "N=10",
        "N=50"
    ],
    "ARWHEAD" => [
        "N=1000",
        "N=5000",
        "N=100",
        "N=500"
    ],
    "BDQRTIC" => [
        "N=1000",
        "N=5000",
        "N=100",
        "N=500"
    ],
    "BOX" => [
        "N=1000",
        "N=100000",
        "N=10000",
        "N=10",
        "N=100"
    ],
    "BOXPOWER" => [
        "N=1000",
        "N=20000",
        "N=10000",
        "N=10",
        "N=100"
    ],
    "BROWNAL" => [
        "N=200",
        "N=100",
        "N=10",
        "N=1000"
    ],
    "BROYDN3DLS" => [
        "KAPPA1=2.0,KAPPA2=1.0,N=1000",
        "KAPPA1=2.0,KAPPA2=1.0,N=500",
        "KAPPA1=2.0,KAPPA2=1.0,N=50",
        "KAPPA1=2.0,KAPPA2=1.0,N=10000",
        "KAPPA1=2.0,KAPPA2=1.0,N=10",
        "KAPPA1=2.0,KAPPA2=1.0,N=5000",
        "KAPPA1=2.0,KAPPA2=1.0,N=100"
    ],
    "BROYDN7D" => [
        "N/2=25",
        "N/2=250",
        "N/2=50",
        "N/2=5",
        "N/2=5000",
        "N/2=2500",
        "N/2=500"
    ],
    "BROYDNBDLS" => [
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=1000,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=500,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=50,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=10000,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=10,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=5000,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=100,UB=1"
    ],
    "BRYBND" => [
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=1000,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=500,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=50,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=10000,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=10,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=5000,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=100,UB=1"
    ],
    "CHAINWOO" => [
        "NS=1",
        "NS=49",
        "NS=499",
        "NS=1999",
        "NS=4999"
    ],
    "CHNROSNB" => [
        "N=25",
        "N=10",
        "N=50"
    ],
    "CHNRSNBM" => [
        "N=25",
        "N=10",
        "N=50"
    ],
    "COSINE" => [
        "N=1000",
        "N=10000",
        "N=100",
        "N=10"
    ],
    "CRAGGLVY" => [
        "M=24",
        "M=4",
        "M=499",
        "M=1",
        "M=2499",
        "M=249",
        "M=49"
    ],
    "CURLY10" => [
        "N=1000",
        "N=10000",
        "N=100"
    ],
    "CURLY20" => [
        "N=1000",
        "N=10000",
        "N=100"
    ],
    "CURLY30" => [
        "N=1000",
        "N=10000",
        "N=100"
    ],
    "DIXMAANA" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANB" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANC" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAAND" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANE" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANF" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANG" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANH" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANI" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANJ" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANK" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANL" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANM" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANN" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANO" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXMAANP" => [
        "M=30",
        "M=3000",
        "M=1000",
        "M=5",
        "M=100",
        "M=500"
    ],
    "DIXON3DQ" => [
        "N=1000",
        "N=10000",
        "N=100",
        "N=10"
    ],
    "DQDRTIC" => [
        "N=1000",
        "N=500",
        "N=50",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "DQRTIC" => [
        "N=1000",
        "N=500",
        "N=50",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "EDENSCH" => [
        "N=2000",
        "N=36"
    ],
    "EIGENALS" => [
        "N=2",
        "N=10",
        "N=50"
    ],
    "EIGENBLS" => [
        "N=2",
        "N=10",
        "N=50"
    ],
    "EIGENCLS" => [
        "M=25",
        "M=2",
        "M=10"
    ],
    "ENGVAL1" => [
        "N=1000",
        "N=50",
        "N=2",
        "N=5000",
        "N=100"
    ],
    "ERRINROS" => [
        "N=25",
        "N=10",
        "N=50"
    ],
    "ERRINRSM" => [
        "N=25",
        "N=10",
        "N=50"
    ],
    "EXTROSNB" => [
        "N=1000",
        "N=100",
        "N=5",
        "N=10"
    ],
    "FLETBV3M" => [
        "KAPPA=0.0,N=1000",
        "KAPPA=0.0,N=10000",
        "KAPPA=0.0,N=10",
        "KAPPA=0.0,N=5000",
        "KAPPA=0.0,N=100",
        "KAPPA=1.0,N=1000",
        "KAPPA=1.0,N=10000",
        "KAPPA=1.0,N=10",
        "KAPPA=1.0,N=5000",
        "KAPPA=1.0,N=100"
    ],
    "FLETCBV2" => [
        "KAPPA=0.0,N=1000",
        "KAPPA=0.0,N=10000",
        "KAPPA=0.0,N=10",
        "KAPPA=0.0,N=5000",
        "KAPPA=0.0,N=100",
        "KAPPA=1.0,N=1000",
        "KAPPA=1.0,N=10000",
        "KAPPA=1.0,N=10",
        "KAPPA=1.0,N=5000",
        "KAPPA=1.0,N=100"
    ],
    "FLETCBV3" => [
        "KAPPA=0.0,N=1000",
        "KAPPA=0.0,N=10000",
        "KAPPA=0.0,N=10",
        "KAPPA=0.0,N=5000",
        "KAPPA=0.0,N=100",
        "KAPPA=1.0,N=1000",
        "KAPPA=1.0,N=10000",
        "KAPPA=1.0,N=10",
        "KAPPA=1.0,N=5000",
        "KAPPA=1.0,N=100"
    ],
    "FLETCHBV" => [
        "KAPPA=0.0,N=1000",
        "KAPPA=0.0,N=10000",
        "KAPPA=0.0,N=10",
        "KAPPA=0.0,N=5000",
        "KAPPA=0.0,N=100",
        "KAPPA=1.0,N=1000",
        "KAPPA=1.0,N=10000",
        "KAPPA=1.0,N=10",
        "KAPPA=1.0,N=5000",
        "KAPPA=1.0,N=100"
    ],
    "FLETCHCR" => [
        "N=1000",
        "N=100",
        "N=10"
    ],
    "FMINSRF2" => [
        "P=4",
        "P=31",
        "P=7",
        "P=75",
        "P=125",
        "P=100",
        "P=11",
        "P=8",
        "P=32"
    ],
    "FMINSURF" => [
        "P=4",
        "P=31",
        "P=7",
        "P=75",
        "P=125",
        "P=100",
        "P=11",
        "P=8",
        "P=32"
    ],
    "FREUROTH" => [
        "N=1000",
        "N=500",
        "N=50",
        "N=2",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "GENHUMPS" => [
        "N=1000,ZETA=20.0",
        "N=1000,ZETA=2.0",
        "N=500,ZETA=20.0",
        "N=500,ZETA=2.0",
        "N=10,ZETA=20.0",
        "N=10,ZETA=2.0",
        "N=5000,ZETA=20.0",
        "N=5000,ZETA=2.0",
        "N=100,ZETA=20.0",
        "N=100,ZETA=2.0",
        "N=5,ZETA=20.0",
        "N=5,ZETA=2.0"
    ],
    "GENROSE" => [
        "N=100",
        "N=500",
        "N=5",
        "N=10"
    ],
    "HILBERTA" => [
        "D=0.0,N=6",
        "D=0.0,N=2",
        "D=0.0,N=4",
        "D=0.0,N=10",
        "D=0.0,N=5"
    ],
    "HILBERTB" => [
        "D=5.0,N=5",
        "D=5.0,N=10",
        "D=5.0,N=50"
    ],
    "INDEF" => [
        "ALPHA=0.5,N=1000",
        "ALPHA=0.5,N=50",
        "ALPHA=0.5,N=10",
        "ALPHA=0.5,N=5000",
        "ALPHA=0.5,N=100",
        "ALPHA=10.0,N=1000",
        "ALPHA=10.0,N=50",
        "ALPHA=10.0,N=10",
        "ALPHA=10.0,N=5000",
        "ALPHA=10.0,N=100",
        "ALPHA=100.0,N=1000",
        "ALPHA=100.0,N=50",
        "ALPHA=100.0,N=10",
        "ALPHA=100.0,N=5000",
        "ALPHA=100.0,N=100",
        "ALPHA=1.0,N=1000",
        "ALPHA=1.0,N=50",
        "ALPHA=1.0,N=10",
        "ALPHA=1.0,N=5000",
        "ALPHA=1.0,N=100",
        "ALPHA=1000.0,N=1000",
        "ALPHA=1000.0,N=50",
        "ALPHA=1000.0,N=10",
        "ALPHA=1000.0,N=5000",
        "ALPHA=1000.0,N=100"
    ],
    "INDEFM" => [
        "ALPHA=0.5,N=1000",
        "ALPHA=0.5,N=100000",
        "ALPHA=0.5,N=50",
        "ALPHA=0.5,N=10000",
        "ALPHA=0.5,N=10",
        "ALPHA=0.5,N=5000",
        "ALPHA=0.5,N=100",
        "ALPHA=10.0,N=1000",
        "ALPHA=10.0,N=100000",
        "ALPHA=10.0,N=50",
        "ALPHA=10.0,N=10000",
        "ALPHA=10.0,N=10",
        "ALPHA=10.0,N=5000",
        "ALPHA=10.0,N=100",
        "ALPHA=100.0,N=1000",
        "ALPHA=100.0,N=100000",
        "ALPHA=100.0,N=50",
        "ALPHA=100.0,N=10000",
        "ALPHA=100.0,N=10",
        "ALPHA=100.0,N=5000",
        "ALPHA=100.0,N=100",
        "ALPHA=1.0,N=1000",
        "ALPHA=1.0,N=100000",
        "ALPHA=1.0,N=50",
        "ALPHA=1.0,N=10000",
        "ALPHA=1.0,N=10",
        "ALPHA=1.0,N=5000",
        "ALPHA=1.0,N=100",
        "ALPHA=1000.0,N=1000",
        "ALPHA=1000.0,N=100000",
        "ALPHA=1000.0,N=50",
        "ALPHA=1000.0,N=10000",
        "ALPHA=1000.0,N=10",
        "ALPHA=1000.0,N=5000",
        "ALPHA=1000.0,N=100"
    ],
    "INTEQNELS" => [
        "N=100",
        "N=500",
        "N=10",
        "N=50"
    ],
    "JIMACK" => [
        "M=2,N=2",
        "M=2,N=12",
        "M=6,N=2",
        "M=6,N=12"
    ],
    "LIARWHD" => [
        "N=1000",
        "N=500",
        "N=10000",
        "N=36",
        "N=5000",
        "N=100"
    ],
    "MANCINO" => [
        "ALPHA=5,BETA=14.0,GAMMA=3,N=50",
        "ALPHA=5,BETA=14.0,GAMMA=3,N=20",
        "ALPHA=5,BETA=14.0,GAMMA=3,N=10",
        "ALPHA=5,BETA=14.0,GAMMA=3,N=30",
        "ALPHA=5,BETA=14.0,GAMMA=3,N=100"
    ],
    "MODBEALE" => [
        "ALPHA=50.0,N/2=1",
        "ALPHA=50.0,N/2=5",
        "ALPHA=50.0,N/2=1000",
        "ALPHA=50.0,N/2=10000",
        "ALPHA=50.0,N/2=2",
        "ALPHA=50.0,N/2=100"
    ],
    "MOREBV" => [
        "N=1000",
        "N=500",
        "N=50",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "MSQRTALS" => [
        "P=70",
        "P=7",
        "P=23",
        "P=2",
        "P=10",
        "P=32"
    ],
    "MSQRTBLS" => [
        "P=70",
        "P=7",
        "P=23",
        "P=3",
        "P=10",
        "P=32"
    ],
    "NCB20" => [
        "N=1000",
        "N=5000",
        "N=100"
    ],
    "NCB20B" => [
        "N=1000",
        "N=500",
        "N=180",
        "N=21",
        "N=50",
        "N=2000",
        "N=22",
        "N=5000",
        "N=100"
    ],
    "NONCVXU2" => [
        "N=1000",
        "N=100000",
        "N=10000",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "NONCVXUN" => [
        "N=1000",
        "N=100000",
        "N=10000",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "NONDIA" => [
        "N=1000",
        "N=90",
        "N=500",
        "N=50",
        "N=20",
        "N=10000",
        "N=10",
        "N=5000",
        "N=30",
        "N=100"
    ],
    "NONDQUAR" => [
        "N=1000",
        "N=500",
        "N=10000",
        "N=5000",
        "N=100"
    ],
    "NONMSQRT" => [
        "P=70",
        "P=7",
        "P=23",
        "P=3",
        "P=10",
        "P=32"
    ],
    "OSCIGRAD" => [
        "N=1000,RHO=500.0",
        "N=1000,RHO=1.0",
        "N=100000,RHO=500.0",
        "N=100000,RHO=1.0",
        "N=15,RHO=500.0",
        "N=15,RHO=1.0",
        "N=2,RHO=500.0",
        "N=2,RHO=1.0",
        "N=10000,RHO=500.0",
        "N=10000,RHO=1.0",
        "N=10,RHO=500.0",
        "N=10,RHO=1.0",
        "N=5,RHO=500.0",
        "N=5,RHO=1.0",
        "N=25,RHO=500.0",
        "N=25,RHO=1.0",
        "N=100,RHO=500.0",
        "N=100,RHO=1.0"
    ],
    "OSCIPATH" => [
        "N=500,RHO=500.0",
        "N=500,RHO=1.0",
        "N=25,RHO=500.0",
        "N=25,RHO=1.0",
        "N=2,RHO=500.0",
        "N=2,RHO=1.0",
        "N=10,RHO=500.0",
        "N=10,RHO=1.0",
        "N=100,RHO=500.0",
        "N=100,RHO=1.0",
        "N=5,RHO=500.0",
        "N=5,RHO=1.0"
    ],
    "PENALTY1" => [
        "N=1000",
        "N=500",
        "N=50",
        "N=4",
        "N=10",
        "N=100"
    ],
    "PENALTY2" => [
        "N=1000",
        "N=500",
        "N=50",
        "N=4",
        "N=10",
        "N=200",
        "N=100"
    ],
    "PENALTY3" => [
        "N/2=25",
        "N/2=50",
        "N/2=100"
    ],
    "POWELLSG" => [
        "N=1000",
        "N=500",
        "N=60",
        "N=20",
        "N=8",
        "N=10000",
        "N=4",
        "N=36",
        "N=16",
        "N=80",
        "N=40",
        "N=5000",
        "N=100"
    ],
    "POWER" => [
        "N=1000",
        "N=500",
        "N=50",
        "N=20",
        "N=10000",
        "N=75",
        "N=10",
        "N=5000",
        "N=30",
        "N=100"
    ],
    "QUARTC" => [
        "N=1000",
        "N=500",
        "N=10000",
        "N=5000",
        "N=100",
        "N=25"
    ],
    "SBRYBND" => [
        "N=1000",
        "N=500",
        "N=50",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "SCHMVETT" => [
        "N=1000",
        "N=500",
        "N=3",
        "N=10000",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "SCOSINE" => [
        "N=1000",
        "N=10000",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "SCURLY10" => [
        "N=1000",
        "N=100000",
        "N=10000",
        "N=1000000",
        "N=10",
        "N=100"
    ],
    "SCURLY20" => [
        "N=1000",
        "N=100000",
        "N=10000",
        "N=1000000",
        "N=10",
        "N=100"
    ],
    "SCURLY30" => [
        "N=1000",
        "N=100000",
        "N=10000",
        "N=1000000",
        "N=10",
        "N=100"
    ],
    "SENSORS" => [
        "N=1000",
        "N=3",
        "N=2",
        "N=10",
        "N=100"
    ],
    "SINQUAD" => [
        "N=1000",
        "N=500",
        "N=50",
        "N=10000",
        "N=5000",
        "N=100",
        "N=5"
    ],
    "SPARSINE" => [
        "N=1000",
        "N=50",
        "N=10000",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "SPARSQUR" => [
        "N=1000",
        "N=50",
        "N=10000",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "SPMSRTLS" => [
        "M=34",
        "M=334",
        "M=3334",
        "M=167",
        "M=10",
        "M=1667"
    ],
    "SROSENBR" => [
        "N/2=25",
        "N/2=250",
        "N/2=50",
        "N/2=5",
        "N/2=5000",
        "N/2=2500",
        "N/2=500"
    ],
    "SSBRYBND" => [
        "N=1000",
        "N=500",
        "N=50",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "SSCOSINE" => [
        "N=1000",
        "N=100000",
        "N=10000",
        "N=1000000",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "TESTQUAD" => [
        "N=1000",
        "N=5000"
    ],
    "TOINTGSS" => [
        "N=1000",
        "N=500",
        "N=50",
        "N=10000",
        "N=10",
        "N=5000",
        "N=100"
    ],
    "TQUARTIC" => [
        "N=1000",
        "N=500",
        "N=50",
        "N=10000",
        "N=10",
        "N=5000",
        "N=100",
        "N=5"
    ],
    "TRIDIA" => [
        "ALPHA=2.0,BETA=1.0,DELTA=1.0,GAMMA=1.0,N=1000",
        "ALPHA=2.0,BETA=1.0,DELTA=1.0,GAMMA=1.0,N=500",
        "ALPHA=2.0,BETA=1.0,DELTA=1.0,GAMMA=1.0,N=50",
        "ALPHA=2.0,BETA=1.0,DELTA=1.0,GAMMA=1.0,N=20",
        "ALPHA=2.0,BETA=1.0,DELTA=1.0,GAMMA=1.0,N=10000",
        "ALPHA=2.0,BETA=1.0,DELTA=1.0,GAMMA=1.0,N=10",
        "ALPHA=2.0,BETA=1.0,DELTA=1.0,GAMMA=1.0,N=5000",
        "ALPHA=2.0,BETA=1.0,DELTA=1.0,GAMMA=1.0,N=30",
        "ALPHA=2.0,BETA=1.0,DELTA=1.0,GAMMA=1.0,N=100"
    ],
    "VARDIM" => [
        "N=200",
        "N=100",
        "N=10",
        "N=50"
    ],
    "VAREIGVL" => [
        "M=4,N=99,Q=1.5",
        "M=4,N=499,Q=1.5",
        "M=4,N=9,Q=1.5",
        "M=4,N=49,Q=1.5",
        "M=4,N=4999,Q=1.5",
        "M=4,N=999,Q=1.5",
        "M=5,N=99,Q=1.5",
        "M=5,N=499,Q=1.5",
        "M=5,N=9,Q=1.5",
        "M=5,N=49,Q=1.5",
        "M=5,N=4999,Q=1.5",
        "M=5,N=999,Q=1.5",
        "M=6,N=99,Q=1.5",
        "M=6,N=499,Q=1.5",
        "M=6,N=9,Q=1.5",
        "M=6,N=49,Q=1.5",
        "M=6,N=4999,Q=1.5",
        "M=6,N=999,Q=1.5"
    ],
    "WATSON" => [
        "N=12",
        "N=31"
    ],
    "WOODS" => [
        "NS=1",
        "NS=1000",
        "NS=2500",
        "NS=25",
        "NS=250"
    ],
    "YATP1LS" => [
        "N=50",
        "N=350",
        "N=10",
        "N=200",
        "N=100"
    ],
    "YATP2LS" => [
        "N=50",
        "N=2",
        "N=350",
        "N=10",
        "N=200",
        "N=100"
    ]
)
fstamp = Dates.format(Dates.now(), dateformat"yyyy/mm/dd HH:MM:SS")
fstamppath = Dates.format(Dates.now(), dateformat"yyyymmddHH")
csvfile = open("cutest-$fstamppath.csv", "w")
write(csvfile, join(header, ","), "\n")
p = Progress(length(PROBLEMS); showspeed=true)
##########################################
# define your filters
# - filter_cutest_problem
##########################################
filter_cutest_problem(nlp) = (4 <= nlp.meta.nvar <= 200)
# filter_optimization_method(k) = k ∉ [:GD]
# filter_optimization_method(k) = k ∉ [:GD]
filter_optimization_method(k) = k == :CG

tables = []
header = ["name", "n", "method", "k", "df", "fx", "t", "status", "update"]
##########################################
# iteration
##########################################
# todo, add field kf, kg, kH, and inner iteration #
for (f, param_combination) in PROBLEMS
    for pc in param_combination
        try
            nlp = CUTEstModel(f, "-param", pc)
            name = "$(nlp.meta.name)-$(nlp.meta.nvar)"

            if !filter_cutest_problem(nlp)
                @warn("problem $name with $(nlp.meta.nvar) is not proper, skip")
                finalize(nlp)
                continue
            end
            @info("problem $name with $(nlp.meta.nvar) $name is good, continue")

            x0 = nlp.meta.x0
            loss(x) = NLPModels.obj(nlp, x)
            g(x) = NLPModels.grad(nlp, x)
            H(x) = NLPModels.hess(nlp, x)

            for (k, v) in OPTIMIZERS
                if !filter_optimization_method(k)
                    continue
                end
                @info("running $name $pc $k")
                line = []
                try
                    r = v(x0, loss, g, H)
                    line = [nlp.meta.name, nlp.meta.nvar, k, r.k, r.state.ϵ, r.state.fx, r.state.t, r.state.ϵ < 1e-5]

                catch e
                    line = [nlp.meta.name, nlp.meta.nvar, k, NaN, NaN, NaN, NaN, false]

                    bt = catch_backtrace()
                    msg = sprint(showerror, e, bt)
                    println(msg)
                    @warn("instance $f opt $k failed")
                end
                # dump
                write(
                    csvfile,
                    join(line, ","),
                    ",",
                    fstamp,
                    "\n"
                )
                flush(csvfile)
            end
            finalize(nlp)

        catch ef
            bt = catch_backtrace()
            msg = sprint(showerror, ef, bt)
            println(msg)
            @warn("instance $f loading failed")
            if isa(ef, InterruptException)
                @warn("user interrupted @ $f")
                exit(code=-1)
            end
        end
        ProgressMeter.next!(p)
        # @comment
        # only play with one that has proper size
        break
    end
end

close(csvfile)