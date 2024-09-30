import math
import numpy as np
import os
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
from io import StringIO
from helper import pd2Gslib, read_gslib_formate

method = "OK"
blur_flag = "blur"
split_size = 0.8
vertical_range = 5
df = read_gslib_formate(welllogpath="./all_samples/W51_por_scale_up").astype(float)
df["z_coord unit1 scale1"] = df["z_coord unit1 scale1"].round(2)
res_points = pd.read_csv(
    "./result/res_{}_zones_{}_{}_{}.txt".format(
        method, str(split_size), str(vertical_range), blur_flag
    ),
    sep=" ",
    header=None,
).astype(float)
res_points.columns = [
    "x_coord unit1 scale1",
    "y_coord unit1 scale1",
    "z_coord unit1 scale1",
    "{}_{}_{}_{}_por".format(method, str(split_size), str(vertical_range), blur_flag),
    "i_index unit1 scale1",
    "j_index unit1 scale1",
    "k_index unit1 scale1",
]
res_points["z_coord unit1 scale1"] = res_points["z_coord unit1 scale1"]
print(res_points.shape, df.shape)
res_points

pd2Gslib(
    res_points,
    df,
    petrel_formate_savepath="./result/res_{}_{}_{}_{}_petrel_Gslibformat.txt".format(
        method, str(split_size), str(vertical_range), blur_flag
    ),
)

print()
