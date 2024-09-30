import numpy as np
import vtk
import os
from numpy.core.fromnumeric import size
import math
import time
import pointsToVTKpoint
import pointsToVTKsurface
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
from concurrent.futures import ProcessPoolExecutor, wait, as_completed
import time
from datetime import datetime
from tqdm import tqdm
from io import StringIO


def call_func_job(args):
    p1 = args[0]
    _points_all = args[1]
    m = []
    for p2 in _points_all:
        if (
            abs(p2[0] - p1[0]) < 45
            and abs(p2[1] - p1[1]) < 45
            and abs(p2[2] - p1[2]) < 6
        ):
            m.append(p2[3])
    point_blur = [p1[0], p1[1], p1[2], sum(m) / len(m), p1[4], p1[5], p1[6]]

    return point_blur


def run(f, this_iter, max_workers=1):
    """
    主函数，运行多进程程序并显示进度条。
    @param f:
    @param this_iter:
    @return:
    """
    process_pool = ProcessPoolExecutor(max_workers)
    results = list(tqdm(process_pool.map(f, this_iter), total=len(this_iter)))
    # print(results)
    process_pool.shutdown()
    return results


def main(txt_path, blur_path):

    vtk_path = blur_path.replace(".txt", ".vtk")

    start2 = time.perf_counter()
    df = pd.read_csv(txt_path, sep=" ", header=None)
    points_all = np.array(df)
    known_point = []
    points_all2 = []
    for p1 in points_all:
        known_point.append(p1)
        points_all2.append(points_all)
        # if len(known_point)>10000:
        #     break
    known_point_points_all = list(zip(known_point, points_all2))
    jobs = run(call_func_job, known_point_points_all, max_workers=30)
    res = []

    for job in jobs:
        res.append(np.array(job))
    res = np.array(res)
    np.savetxt(blur_path, res, delimiter=" ", fmt="%f")
    # pointsToVTKpoint.txt_to_vtk(blur_path, vtk_path)
    end2 = time.perf_counter()
    print("Blur  running time all: %s Seconds" % (end2 - start2))


if __name__ == "__main__":
    txt_path = "res_IDW_zones_2602.txt"
    blur_path = txt_path.replace(".txt", "_blur.txt")

    main(txt_path, blur_path)
