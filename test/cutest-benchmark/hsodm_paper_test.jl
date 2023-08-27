
# unconstrained problems 2022/11/04
# I select a set of unconstrained problems
TEST = Dict(
    "EXTROSNB" => ["N=100"]
)
UNC_PROBLEMS_4to200 = Dict(
    "ARGLINA" => ["M=200,N=200"], "ARGLINB" => ["M=200,N=200"],
    "ARGLINC" => ["M=200,N=200"], "ARGTRIGLS" => ["N=200"],
    "ARWHEAD" => ["N=100"], "BDQRTIC" => ["N=100"], "BOX" => ["N=10"], "BOXPOWER" => ["N=10"], "BROWNAL" => ["N=200"], "BROYDN3DLS" => ["KAPPA1=2.0,KAPPA2=1.0,N=50"], "BROYDN7D" => ["N/2=25"], "BROYDNBDLS" => ["KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=50,UB=1"], "BRYBND" => ["KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=50,UB=1"], "CHAINWOO" => ["NS=1"], "CHNROSNB" => ["N=25"], "CHNRSNBM" => ["N=25"], "COSINE" => ["N=100"], "CRAGGLVY" => ["M=24"], "CURLY10" => ["N=100"], "CURLY20" => ["N=100"], "DIXMAANA" => ["M=30"], "DIXMAANB" => ["M=30"], "DIXMAANC" => ["M=30"], "DIXMAAND" => ["M=30"], "DIXMAANE" => ["M=30"], "DIXMAANF" => ["M=30"], "DIXMAANG" => ["M=30"], "DIXMAANH" => ["M=30"], "DIXMAANI" => ["M=30"], "DIXMAANJ" => ["M=30"], "DIXMAANK" => ["M=30"], "DIXMAANL" => ["M=30"], "DIXMAANM" => ["M=30"], "DIXMAANN" => ["M=30"], "DIXMAANO" => ["M=30"], "DIXMAANP" => ["M=30"], "DIXON3DQ" => ["N=100"], "DQDRTIC" => ["N=50"], "DQRTIC" => ["N=50"], "EDENSCH" => ["N=36"], "EIGENALS" => ["N=2"], "EIGENBLS" => ["N=2"], "EIGENCLS" => ["M=2"], "ENGVAL1" => ["N=50"], "ERRINROS" => ["N=25"], "ERRINRSM" => ["N=25"], "EXTROSNB" => ["N=100"], "FLETBV3M" => ["KAPPA=0.0,N=10"], "FLETCBV2" => ["KAPPA=0.0,N=10"], "FLETCBV3" => ["KAPPA=0.0,N=10"], "FLETCHBV" => ["KAPPA=0.0,N=10"], "FLETCHCR" => ["N=100"], "FMINSRF2" => ["P=4"], "FMINSURF" => ["P=4"], "FREUROTH" => ["N=50"], "GENHUMPS" => ["N=10,ZETA=20.0"], "GENROSE" => ["N=100"], "HILBERTA" => ["D=0.0,N=6"], "HILBERTB" => ["D=5.0,N=5"], "INDEF" => ["ALPHA=0.5,N=50"], "INDEFM" => ["ALPHA=0.5,N=50"], "INTEQNELS" => ["N=100"], "JIMACK" => ["M=2,N=2"], "LIARWHD" => ["N=36"], "MANCINO" => ["ALPHA=5,BETA=14.0,GAMMA=3,N=50"], "MODBEALE" => ["ALPHA=50.0,N/2=5"], "MOREBV" => ["N=50"], "MSQRTALS" => ["P=7"], "MSQRTBLS" => ["P=7"], "NCB20" => ["N=100"], "NCB20B" => ["N=180"], "NONCVXU2" => ["N=10"], "NONCVXUN" => ["N=10"], "NONDIA" => ["N=90"], "NONDQUAR" => ["N=100"], "NONMSQRT" => ["P=7"], "OSCIGRAD" => ["N=15,RHO=500.0"], "OSCIPATH" => ["N=25,RHO=500.0"], "PENALTY1" => ["N=50"], "PENALTY2" => ["N=50"], "PENALTY3" => ["N/2=25"], "POWELLSG" => ["N=60"], "POWER" => ["N=50"], "QUARTC" => ["N=100"], "SBRYBND" => ["N=50"], "SCHMVETT" => ["N=10"], "SCOSINE" => ["N=10"], "SCURLY10" => ["N=10"], "SENSORS" => ["N=10"], "SINQUAD" => ["N=50"], "SPARSINE" => ["N=50"], "SPARSQUR" => ["N=50"], "SPMSRTLS" => ["M=34"], "SROSENBR" => ["N/2=25"], "SSBRYBND" => ["N=50"], "SSCOSINE" => ["N=10"], "TOINTGSS" => ["N=50"], "TQUARTIC" => ["N=50"], "TRIDIA" => ["ALPHA=2.0,BETA=1.0,DELTA=1.0,GAMMA=1.0,N=50"], "VARDIM" => ["N=200"], "VAREIGVL" => ["M=4,N=99,Q=1.5"], "WATSON" => ["N=12"], "WOODS" => ["NS=1"], "YATP1LS" => ["N=10"], "YATP2LS" => ["N=2"]
)
UNC_PROBLEMS_200to5000 = Dict("ARWHEAD" => ["N=1000"], "BDQRTIC" => ["N=1000"], "BOX" => ["N=1000"], "BOXPOWER" => ["N=1000"], "BROWNAL" => ["N=1000"], "BROYDN3DLS" => ["KAPPA1=2.0,KAPPA2=1.0,N=1000"], "BROYDN7D" => ["N/2=250"], "BROYDNBDLS" => ["KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=1000,UB=1"], "BRYBND" => ["KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=1000,UB=1"], "CHAINWOO" => ["NS=499"], "COSINE" => ["N=1000"], "CRAGGLVY" => ["M=499"], "CURLY10" => ["N=1000"], "CURLY20" => ["N=1000"], "CURLY30" => ["N=1000"], "DIXMAANA" => ["M=1000"], "DIXMAANB" => ["M=1000"], "DIXMAANC" => ["M=1000"], "DIXMAAND" => ["M=1000"], "DIXMAANE" => ["M=1000"], "DIXMAANF" => ["M=1000"], "DIXMAANG" => ["M=1000"], "DIXMAANH" => ["M=1000"], "DIXMAANI" => ["M=1000"], "DIXMAANJ" => ["M=1000"], "DIXMAANK" => ["M=1000"], "DIXMAANL" => ["M=1000"], "DIXMAANM" => ["M=1000"], "DIXMAANN" => ["M=1000"], "DIXMAANO" => ["M=1000"], "DIXMAANP" => ["M=1000"], "DIXON3DQ" => ["N=1000"], "DQDRTIC" => ["N=1000"], "DQRTIC" => ["N=1000"], "EDENSCH" => ["N=2000"], "EIGENALS" => ["N=50"], "EIGENBLS" => ["N=50"], "EIGENCLS" => ["M=25"], "ENGVAL1" => ["N=1000"], "EXTROSNB" => ["N=1000"], "FLETBV3M" => ["KAPPA=0.0,N=1000"], "FLETCBV2" => ["KAPPA=0.0,N=1000"], "FLETCBV3" => ["KAPPA=0.0,N=1000"], "FLETCHBV" => ["KAPPA=0.0,N=1000"], "FLETCHCR" => ["N=1000"], "FMINSRF2" => ["P=31"], "FMINSURF" => ["P=31"], "FREUROTH" => ["N=1000"], "GENHUMPS" => ["N=1000,ZETA=20.0"], "GENROSE" => ["N=500"], "INDEF" => ["ALPHA=0.5,N=1000"], "INDEFM" => ["ALPHA=0.5,N=1000"], "INTEQNELS" => ["N=500"], "JIMACK" => ["M=2,N=12"], "LIARWHD" => ["N=1000"], "MODBEALE" => ["ALPHA=50.0,N/2=1000"], "MOREBV" => ["N=1000"], "MSQRTALS" => ["P=70"], "MSQRTBLS" => ["P=70"], "NCB20" => ["N=1000"], "NCB20B" => ["N=1000"], "NONCVXU2" => ["N=1000"], "NONCVXUN" => ["N=1000"], "NONDIA" => ["N=1000"], "NONDQUAR" => ["N=1000"], "NONMSQRT" => ["P=70"], "OSCIGRAD" => ["N=1000,RHO=500.0"], "OSCIPATH" => ["N=500,RHO=500.0"], "PENALTY1" => ["N=1000"], "PENALTY2" => ["N=1000"], "POWELLSG" => ["N=1000"], "POWER" => ["N=1000"], "QUARTC" => ["N=1000"], "SBRYBND" => ["N=1000"], "SCHMVETT" => ["N=1000"], "SCOSINE" => ["N=1000"], "SCURLY10" => ["N=1000"], "SCURLY20" => ["N=1000"], "SCURLY30" => ["N=1000"], "SENSORS" => ["N=1000"], "SINQUAD" => ["N=1000"], "SPARSINE" => ["N=1000"], "SPARSQUR" => ["N=1000"], "SPMSRTLS" => ["M=334"], "SROSENBR" => ["N/2=250"], "SSBRYBND" => ["N=1000"], "SSCOSINE" => ["N=1000"], "TESTQUAD" => ["N=1000"], "TOINTGSS" => ["N=1000"], "TQUARTIC" => ["N=1000"], "TRIDIA" => ["ALPHA=2.0,BETA=1.0,DELTA=1.0,GAMMA=1.0,N=1000"], "VAREIGVL" => ["M=4,N=499,Q=1.5"], "WOODS" => ["NS=1000"], "YATP1LS" => ["N=50"], "YATP2LS" => ["N=50"])



UNC_PROBLEMS_221104 = Dict(
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
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=50,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=1000,UB=1",
        "KAPPA1=2.0,KAPPA2=5.0,KAPPA3=1.0,LB=5,N=500,UB=1",
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
        "N=10",
        "N=100"
    ],
    "SCURLY20" => [
        "N=1000",
        "N=10",
        "N=100"
    ],
    "SCURLY30" => [
        "N=1000",
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


