import seaborn as sns

DATES = [
    "2025-06-01",
    "2025-06-05-a",
    "2025-06-05-b",
    "2025-06-05-c",
    "2025-06-02",
    "2025-06-06-a",
    "2025-06-06-b",
    "2025-06-06-c",
    "2025-06-07-a",
    "2025-06-07-b",
    "2025-06-07-c",
    "2025-07-01",
    "2025-07-02",
    "2025-07-03",
    "2025-07-04",
    "2025-07-05-a",
    "2025-07-06-a",
    "2025-07-07-a",
    "2025-07-08-a",
    "2025-07-09-a",
    "2025-07-51",
    "2025-07-52",
    "2025-07-53",
    "2025-07-54",
    "2025-07-55",
    "2025-07-56-a",
    "2025-07-57",
    "2025-07-58-b",
    "2025-07-59-a",
    "2025-07-60-a",
]

TITLE = {
    "2025-06-02": "Regression $N(0, 1)$",
    "2025-06-07-a": "Regression $t_5$",
    "2025-06-07-b": "Regression $t_4$",
    "2025-06-07-c": "Regression $t_3$",
    "2025-06-06-a": "Regression Dept. $s_1$",
    "2025-06-06-b": "Regression Dept. $s_2$",
    "2025-06-06-c": "Regression Dept. $s_3$",
    "2025-06-01": "Classification Logistic",
    "2025-06-05-a": "Classification GMM $a=0$",
    "2025-06-05-b": "Classification GMM $a=-1$",
    "2025-06-05-c": "Classification GMM $a=-2$",
    "2025-07-04": "concrete",
    "2025-07-01": "quake",
    "2025-07-02": "airfoil",
    "2025-07-05-a": "energy",
    "2025-07-08-a": "fish",
    "2025-07-03": "kin8nm",
    "2025-07-09-a": "auction",
    "2025-07-06-a": "grid",
    "2025-07-07-a": "abalone",
    "2025-07-54": "rice",
    "2025-07-57": "sepsis",
    "2025-07-60-a": "banknote",
    "2025-07-55": "mozilla",
    "2025-07-53": "skin",
    "2025-07-51": "blood",
    "2025-07-52": "phoneme",
    "2025-07-56-a": "telescope",
    "2025-07-58-b": "yeast",
    "2025-07-59-a": "wine",
}

# Colour palette for each method
palette = sns.color_palette("colorblind")
COLOR_PALETTE = {
    "tabpfn": palette[0],
    "bb": palette[1],
    "gibbs-eb": palette[3],
    "copula": palette[2],
}

MARKER_SHAPES = {
    "tabpfn": "o",
    "bb": "X",
    "gibbs-eb": "d",
    "copula": "^",
}



POSTERIOR_NAMES = {
    "tabpfn": "TabMGP",
    "bb": "BB",
    "gibbs-eb": "Bayes",
    "copula": "Copula",
}
