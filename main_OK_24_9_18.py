from concurrent import futures
import time
import random
from concurrent.futures import ProcessPoolExecutor

import os
import pandas as pd
from decimal import Decimal
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import numpy as np
from math import *
from numpy.linalg import *
import math
import os

# import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from pykrige.ok3d import OrdinaryKriging3D
from concurrent.futures import ProcessPoolExecutor, wait, as_completed
import time
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import (
    QuantileTransformer,
)  # 导入sklearn 分位数转换器 将双峰数据正态化
from scipy import stats
from scipy.special import inv_boxcox
import itertools


def OK3D(unknownpoint, rangeknownpoints):

    if rangeknownpoints.shape[0] < 3 or np.std(rangeknownpoints[:, 3]) < 0.001:
        # print(unknownpoint)
        res = unknownpoint.tolist()[0][:3]
        res.append(rangeknownpoints[:, 3].mean())

        return list(itertools.chain(res, unknownpoint.tolist()[0][3:]))

    # rangeknownpoints_normal, lamb = stats.boxcox(
    #     np.array(args[1][:, 3])
    # )  # 对数据进行BOX-COX变换
    ok3d = OrdinaryKriging3D(
        rangeknownpoints[:, 0],
        rangeknownpoints[:, 1],
        rangeknownpoints[:, 2],
        rangeknownpoints[:, 3].reshape(-1),
        variogram_model="spherical",
    )
    k3d1 = ok3d.execute(
        "points", unknownpoint[0][0], unknownpoint[0][1], unknownpoint[0][2]
    )
    v = k3d1[0][0]
    if math.isnan(v):
        v = np.mean(rangeknownpoints[:, 3])
    res = unknownpoint.tolist()[0][:3]
    res.append(v)
    return list(itertools.chain(res, unknownpoint.tolist()[0][3:]))


def run(f, this_iter, max_workers=32):
    process_pool = ProcessPoolExecutor(max_workers)
    results = list(tqdm(process_pool.map(f, this_iter), total=len(this_iter)))
    process_pool.shutdown()
    return results


def gridsearchcv(rangeknownpoints):
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut()
    X = rangeknownpoints[:, 0:3]
    y = rangeknownpoints[:, 3:]
    y_pre = []
    y_real = []
    for train_index, test_index in loo.split(X):
        # 分割数据
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        unknown = np.hstack((X_test, y_test))
        rangeknown = np.hstack((X_train, y_train))
        pre = OK3D(X_test, rangeknown)[3]

        y_real.append(y_test[0][0])
        y_pre.append(pre)
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(np.array(y_real) / 100, np.array(y_pre) / 100)
    mse = np.round(mse, 4)

    with open("UQ_OK.txt", "a") as f:
        f.writelines(str(mse) + "\n")


def call_func_job(args):
    unknownpoint = np.array([args[0]])
    rangeknownpoints = np.empty(shape=(args[1].shape[0], 4))
    rangeknownpoints[:, 0] = args[1][:, 0]
    rangeknownpoints[:, 1] = args[1][:, 1]
    rangeknownpoints[:, 2] = args[1][:, 2]
    rangeknownpoints[:, 3] = args[1][:, 3]
    # gridsearchcv(rangeknownpoints)
    res = OK3D(unknownpoint, rangeknownpoints)
    return res


def find_near_points(
    knownpoints_df, unknownpoints_df, major_range, minor_range, vertical_range
):
    outrangegrid = []
    unknownpoint_all = []
    rangeknownpoints_all = []
    for idx, unknownpoint in unknownpoints_df.iterrows():
        unknownpointx = unknownpoint[3]
        unknownpointy = unknownpoint[4]
        unknownpointz = unknownpoint[5]
        unknownpoint = []

        flag = ((knownpoints_df.iloc[:, 3] - unknownpointx) ** 2) / major_range**2 + (
            (knownpoints_df.iloc[:, 4] - unknownpointy) ** 2
        ) / minor_range**2 + (
            (knownpoints_df.iloc[:, 5] - unknownpointz) ** 2
        ) / vertical_range**2 <= 1
        near_points = np.array(
            knownpoints_df[flag == True][
                [
                    "x_coord unit1 scale1",
                    "y_coord unit1 scale1",
                    "z_coord unit1 scale1",
                    "S2X4_POR unit1 scale1",
                    "S2X4_FACIES unit1 scale1",
                ]
            ]
        )
        if near_points.shape[0] == 0:
            outrangegrid.append(
                np.array([unknownpointx, unknownpointy, unknownpointz, 0])
            )
        elif near_points.shape[0] < 4:
            outrangegrid.append(
                np.array(
                    [
                        unknownpointx,
                        unknownpointy,
                        unknownpointz,
                        near_points[:, 3].mean(),
                    ]
                )
            )
        elif near_points.shape[0] > 3:
            rangeknownpoints_all.append(near_points)
            unknownpoint_all.append([unknownpointx, unknownpointy, unknownpointz])

        print(
            "find nearpoints process {:.2f}%".format(
                (idx / unknownpoints_df.shape[0]) * 100
            )
        )

    # unknownpoint_all=unknownpoint_all[300:700]
    # rangeknownpoints_all=rangeknownpoints_all[300:700]
    unknownpoints_rangeknownpoints = list(zip(unknownpoint_all, rangeknownpoints_all))
    return unknownpoints_rangeknownpoints, outrangegrid


if __name__ == "__main__":

    split_size = 0.8
    major_range = 2602  # 2602.759
    minor_range = 2286  # 2286.249
    vertical_range = 5  # 73.04
    save_res_path = "./result/res_OK_zones_{}_{}.txt".format(
        str(split_size), str(str(vertical_range))
    )
    knownpoints_df = pd.read_csv(
        "./train/known_points_train_{}.csv".format(str(split_size))
    )
    unknownpoints_df = pd.read_csv("./all_samples/unknown_points_grid.csv")

    unknownpoints_rangeknownpoints, outrangegrid = find_near_points(
        knownpoints_df, unknownpoints_df, major_range, minor_range, vertical_range
    )

    import helper


    helper.save_unknownpoints_rangeknownpoints(
        unknownpoints_rangeknownpoints,
        outrangegrid,
        "./train/unknownpoints_rangeknownpoints_{}_{}.json".format(
            str(split_size), str(vertical_range)
        ),
        "./train/outrangegrid_{}_{}.txt".format(str(split_size), str(vertical_range)),
    )
    unknownpoints_rangeknownpoints_zipiter, outrangegrid = (
        helper.read_unknownpoints_rangeknownpoints(
            "./train/unknownpoints_rangeknownpoints_{}_{}.json".format(
                str(split_size), str(vertical_range)
            ),
            "./train/outrangegrid_{}_{}.txt".format(
                str(split_size), str(vertical_range)
            ),
        )
    )
    # unknownpoints_rangeknownpoints_zipiter = unknownpoints_rangeknownpoints_zipiter[
    #     :100
    # ]
    print(len(unknownpoints_rangeknownpoints_zipiter))
    start2 = time.perf_counter()
    jobs = run(call_func_job, unknownpoints_rangeknownpoints_zipiter, max_workers=4)

    unknowngrids_res = outrangegrid

    for job in jobs:

        unknowngrids_res.append(job)

    with open(save_res_path, "w") as f:
        for idx, i in enumerate(unknowngrids_res):
            if str(i[3]) == "nan":
                i[3] = 0
            f.write(
                str(i[0])
                + " "
                + str(i[1])
                + " "
                + str(i[2])
                + " "
                + str(i[3])
                + " "
                + str(i[4])
                + " "
                + str(i[5])
                + " "
                + str(i[6])
                + "\n"
            )
    end2 = time.perf_counter()
    print("Running time all: %s Seconds" % (end2 - start2))
    import Blurhelper

    blur_path = save_res_path.replace(".txt", "_blur.txt")
    Blurhelper.main(save_res_path, blur_path)

    # import pointsToVTKpoint
    # vtk_path = blur_path.replace(".txt", ".vtk")
    # pointsToVTKpoint.txt_to_vtk(blur_path, vtk_path)
