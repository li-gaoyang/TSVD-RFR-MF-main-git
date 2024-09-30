import numpy as np
import json
import math
import numpy as np
import os
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
from io import StringIO


def save_unknownpoints_rangeknownpoints(
    unknownpoints_rangeknownpoints, outrangegrid, path0, path1
):
    data_dict = {}
    for v in unknownpoints_rangeknownpoints:
        v0 = json.dumps(v[0])
        v1 = np.array2string(v[1], separator=", ")
        data_dict[v0] = v1
    json_string = json.dumps(data_dict)
    # 将字符串写入文本文件
    with open(path0, "w") as file:
        file.write(json_string)

    np.savetxt(path1, np.array(outrangegrid), delimiter=",", fmt="%d")


def read_unknownpoints_rangeknownpoints(path0, path1):
    json_string = ""
    with open(path0, "r") as file:
        json_string = file.read()
    json_dict = json.loads(json_string)
    unknownpoint_all = []
    rangeknownpoints_all = []

    for key, value in json_dict.items():
        unknownpoint_all.append(json.loads(key))
        rangeknownpoints_all
        # 移除方括号和换行符
        cleaned_string = (
            value.replace("[", "").replace("]", "").replace(" ", "").replace("\n", "")
        )
        array_from_str = np.fromstring(cleaned_string, dtype=float, sep=",")

        # 将字符串转换回numpy数组
        data_array = array_from_str.reshape(int(array_from_str.shape[0] / 5), 5)
        rangeknownpoints_all.append(data_array)

    unknownpoints_rangeknownpoints_zipiter = list(
        zip(unknownpoint_all, rangeknownpoints_all)
    )
    outrangegrid = np.loadtxt(path1, delimiter=",", dtype=float)

    return unknownpoints_rangeknownpoints_zipiter, outrangegrid.tolist()


def pd2Gslib(S2X4_welllog_por2, df_all, petrel_formate_savepath="petrel_formate.txt"):
    df = df_all
    df1 = df[
        [
            "i_index unit1 scale1",
            "j_index unit1 scale1",
            "k_index unit1 scale1",
            "x_coord unit1 scale1",
            "y_coord unit1 scale1",
            "z_coord unit1 scale1",
        ]
    ]

    merged_df = pd.merge(
        df1,
        S2X4_welllog_por2,
        on=["i_index unit1 scale1", "j_index unit1 scale1", "k_index unit1 scale1"],
        how="outer",
    )
    merged_df = merged_df.fillna(-99)
    merged_df = merged_df[
        [
            "i_index unit1 scale1",
            "j_index unit1 scale1",
            "k_index unit1 scale1",
            "x_coord unit1 scale1_x",
            "y_coord unit1 scale1_x",
            "z_coord unit1 scale1_x",
            S2X4_welllog_por2.columns[3],
        ]
    ]
    merged_df.columns = [
        "i_index unit1 scale1",
        "j_index unit1 scale1",
        "k_index unit1 scale1",
        "x_coord unit1 scale1",
        "y_coord unit1 scale1",
        "z_coord unit1 scale1",
        S2X4_welllog_por2.columns[3],
    ]

    merged_df.to_csv(petrel_formate_savepath, sep=" ", index=False, header=False)
    head_list = merged_df.columns.tolist()
    head_list.insert(0, str(len(merged_df.columns.tolist())))
    head_list.insert(0, "PETREL: Properties")
    head_str = "\n".join(head_list)
    str_txt = head_str + "\n"
    with open(petrel_formate_savepath, "r", encoding="utf-8") as file:
        content = file.read()
        str_txt = str_txt + content
    with open(petrel_formate_savepath, "w", encoding="utf-8") as file:
        file.write(str_txt)


def read_gslib_formate(welllogpath="W51_por_scale_up"):
    content = ""
    heads = []
    with open(welllogpath, "r", encoding="utf-8") as file:
        idx = 0
        headnum = 9999
        heads = []
        bodys = ""
        for line in file:
            if idx == 1:
                headnum = int(line.strip())
            if idx > 1 and idx < headnum + 2:
                heads.append(line.strip())
            idx = idx + 1
    with open(welllogpath, "r", encoding="utf-8") as file:
        split_str = heads[len(heads) - 1]
        content = file.read()
        contents = content.split(split_str)
        bodys = contents[1][1:]
    df = pd.read_csv(StringIO(bodys), header=None, sep=" ")
    df = df.iloc[:, :-1]
    df = df.apply(pd.to_numeric, errors="ignore")
    df.columns = heads
    return df


def create_model_score(model_score_path):
    if os.path.exists(model_score_path):
        os.remove(model_score_path)
    else:
        with open(model_score_path, "w") as file:
            pass
