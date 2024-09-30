import numpy as np
import vtk
import os
from numpy.core.fromnumeric import size
import math
import random
from sklearn import ensemble
from sklearn.decomposition import TruncatedSVD
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import math
from sklearn.model_selection import GridSearchCV
import warnings

# 过滤掉RuntimeWarning警告
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Calculate distance
def distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


# Calculate inclination
def cal_incl(x0, y0, z0, x1, y1, z1):
    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1
    t = pow(pow(dx, 2) + pow(dy, 2), 0.5)
    incl = 0
    if t == 0:
        incl = 0
    else:
        incl = math.atan(dz / t)
    return incl


# Calculate azimuth
def cal_azim(x1, y1, x2, y2):
    theta_radians = math.atan2(y1 - y2, x1 - x2)
    theta_degrees = math.degrees(theta_radians)
    if theta_degrees < 0:
        theta_degrees += 360
    return theta_degrees


# make TSVD-RFR  interpolation model
# "xyz0" is the coordinate of the unknown point
# "xyz_nei" is some known sample points near unknown points
# return is the attribute value of xyz0 and the coordinates of xyz0
# You can find the input format and output format of data in the "__main__" function.
def RandomForestInterpolation3D(
    unknownpoints, rangeknownpoints, other_factors, n_jobs=1
):
    # print(unknownpoints,np.array(rangeknownpoints).shape)
    # if unknownpoints[0][0]==48060 and   unknownpoints[0][1]==50060    and   unknownpoints[0][2]== -2864.26950073:
    #     print(111)
    train_x = []
    train_y = []
    # 假设未知点，建立模型
    for idx, assumingunknownpoint in enumerate(rangeknownpoints):  # 遍历每个已知点

        x = assumingunknownpoint[0]
        y = assumingunknownpoint[1]
        z = assumingunknownpoint[2]
        v = assumingunknownpoint[3]
        # print(x, y, z)
        #  get train dataset as the input of model, the number of variable is n-1
        _train_xs = []
        for i in range(len(rangeknownpoints)):
            if (
                abs(x - rangeknownpoints[i][0]) < 0.00001
                and abs(y - rangeknownpoints[i][1]) < 0.00001
                and abs(z - rangeknownpoints[i][2]) < 0.00001
            ):
                aaa = 1
                # with open("log2.txt","a") as f:
                #     f.writelines(str(idx)+'==================='+str(np.array(assumingunknownpoint)))
                #     f.writelines("\n")
                # print(idx,assumingunknownpoint,rangeknownpoints[i])
            else:
                _v = float(rangeknownpoints[i][3])  # 获取孔隙度值
                _face = float(other_factors[i])  # 获取沉积相
                dis = distance(
                    x,
                    y,
                    z,
                    rangeknownpoints[i][0],
                    rangeknownpoints[i][1],
                    rangeknownpoints[i][2],
                )  # 获取假设未知点到已知点的距离
                azim = cal_azim(
                    x, y, rangeknownpoints[i][0], rangeknownpoints[i][1]
                )  # 获取假设未知点与已知点的方位角
                incl = cal_incl(
                    x,
                    y,
                    z,
                    rangeknownpoints[i][0],
                    rangeknownpoints[i][1],
                    rangeknownpoints[i][2],
                )  # 获取假设未知点与已知点的倾斜角
                _train_x = [_v, _face, dis, azim, incl]
                _train_xs.append(
                    _train_x
                )  # 假设未知点对应的已知点的特征,特征是根据距离远近排序
                if len(rangeknownpoints) - 2 == len(_train_xs):
                    break

        # print(len(_train_xs))
        tr_x = np.array(_train_xs)  # 假设未知点对应的已知点的特征
        tsvd = TruncatedSVD(n_components=1)  # tsvd
        X_tsvd = tsvd.fit_transform(tr_x)
        X_tsvd = X_tsvd.reshape(-1)
        t1 = X_tsvd.tolist()
        unknownpoint = np.array(t1)  # 假设的未知点特征降维

        train_x.append(unknownpoint)
        train_y.append(v)  # 假设的未知点的值

    train_x = np.array(train_x)
    # print(train_x.shape)
    # tsvd2 = TruncatedSVD(n_components=1)  # tsvd
    # train_x = tsvd2.fit_transform(train_x)

    train_y = np.array(train_y)

    model_RFR = ensemble.RandomForestRegressor(n_jobs=n_jobs)

    # #Finding the optimal parameters
    # param_grid = [
    #     {'n_estimators': [ 10, 100], 'max_features': [10, 100]},
    #     {'bootstrap': [True], 'n_estimators': [10, 100], 'max_features': [ 10, 100]},
    # ]
    # cv_num = 5  #Cross validation parameters
    # if len(train_y) < 6:
    #     cv_num = 2

    #     # 使用'R2'作为评分标准，进行交叉验证
    # grid_search = GridSearchCV(estimator=model_RFR, param_grid=param_grid, scoring='r2', cv=cv_num, n_jobs=1)

    # grid_search.fit(train_x, train_y)
    # test_score = grid_search.best_score_

    # # 使用最佳参数的模型在测试集上进行评估
    # best_model = grid_search.best_estimator_
    # test_score = best_model.score(train_x, train_y)
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import make_scorer, mean_squared_error

    loo = LeaveOneOut()
    mse_scorer = make_scorer(mean_squared_error)
    rf = RandomForestRegressor()
    scores = cross_val_score(rf, train_x, train_y / 100, cv=loo, scoring=mse_scorer)
    with open("UQ_RFR.txt", "a") as f:
        f.writelines(str(scores.mean()) + "\n")

    # fit model
    model_RFR.fit(train_x, train_y)
    best_score = model_RFR.score(train_x, train_y)

    # print(best_score)
    with open("model_score.txt", "a") as f:
        f.writelines(
            str(train_x.shape[0])
            + " "
            + str(train_x.shape[1])
            + " "
            + str(best_score)
            + "\n"
        )

    test_x = []
    # 真实未知点
    for unknownpoint in unknownpoints:
        x = unknownpoint[0]
        y = unknownpoint[1]
        z = unknownpoint[2]

        #  get train dataset as the input of model, the number of variable is n-1
        _test_xs = []
        for i in range(len(rangeknownpoints) - 1):  # 获取未知点附近的n个已知点
            _v = float(rangeknownpoints[i][3])  # 获取孔隙度值
            _face = float(other_factors[i])  # 获取沉积相
            dis = distance(
                x,
                y,
                z,
                rangeknownpoints[i][0],
                rangeknownpoints[i][1],
                rangeknownpoints[i][2],
            )  # 获取假设未知点到已知点的距离
            azim = cal_azim(
                x, y, rangeknownpoints[i][0], rangeknownpoints[i][1]
            )  # 获取假设未知点与已知点的方位角
            incl = cal_incl(
                x,
                y,
                z,
                rangeknownpoints[i][0],
                rangeknownpoints[i][1],
                rangeknownpoints[i][2],
            )  # 获取假设未知点与已知点的倾斜角
            _test_x = [_v, _face, dis, azim, incl]
            _test_xs.append(_test_x)  # 假设未知点对应的已知点的特征
            if len(rangeknownpoints) - 2 == len(_test_xs):
                break
        tr_x = np.array(_test_xs)  # 假设未知点对应的已知点的特征
        tsvd = TruncatedSVD(n_components=1)  # tsvd
        X_tsvd = tsvd.fit_transform(tr_x)
        X_tsvd = X_tsvd.reshape(-1)
        t1 = X_tsvd.tolist()
        unknownpointfeatures = np.array(t1)
        test_x.append(unknownpointfeatures)

    # RandomForestRegressor
    starttime = time.perf_counter()

    test_x = np.array(test_x)
    # tsvd3 = TruncatedSVD(n_components=1)  # tsvd
    # test_x = tsvd3.fit_transform(test_x)
    test_y = model_RFR.predict(test_x)

    # print(best_score)
    with open("model_average_coverage_probability.txt", "a") as f:
        f.writelines(
            str(rangeknownpoints[:, 3].min())
            + " "
            + str(rangeknownpoints[:, 3].max())
            + " "
            + str(test_y[0])
            + "\n"
        )

    endtime = time.perf_counter()

    return np.insert(unknownpoints, unknownpoints.shape[1], test_y, axis=1)


# result write to txt


# Read sample points from txt


if __name__ == "__main__":
    print()
    # xyz0 = [0, 0, 0]  # Coordinates of unknownpoint
    # xyz_nei = [[5, 9, 0, 0.11,0], [5, 9, 1, 0.11,0], [5, 9, 2, 0.1,1], [5, 9, 3, 0.1,2], [
    #     5, 9, 4, 0.1,3], [5, 10, 6, 0.3,3],[4, 8, 0, 0.13,2]]  # Coordinates and values of five known sample points
    # res = RandomForestInterpolation3D(xyz0, xyz_nei)
    # print(res)
