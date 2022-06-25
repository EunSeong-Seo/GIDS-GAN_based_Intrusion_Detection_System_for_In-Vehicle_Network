import numpy as np
import pandas as pd


def one_hot_vector(c):
    # convert element of can_data to one hot vector to select can image position
    #        order: 1  2  3  4  5  6  7  8  9  0  a  b  c  d  e  f
    ohv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    if c == "1":
        ohv[0] = 1
    elif c == "2":
        ohv[1] = 1
    elif c == "3":
        ohv[2] = 1
    elif c == "4":
        ohv[3] = 1
    elif c == "5":
        ohv[4] = 1
    elif c == "6":
        ohv[5] = 1
    elif c == "7":
        ohv[6] = 1
    elif c == "8":
        ohv[7] = 1
    elif c == "9":
        ohv[8] = 1
    elif c == "0":
        ohv[9] = 1
    elif c == "a":
        ohv[10] = 1
    elif c == "b":
        ohv[11] = 1
    elif c == "c":
        ohv[12] = 1
    elif c == "d":
        ohv[13] = 1
    elif c == "e":
        ohv[14] = 1
    elif c == "f":
        ohv[15] = 1
    return ohv


def convert_to_can_data_to_image(can):
    # convert can_data to image

    c1 = one_hot_vector(can[0])
    c2 = one_hot_vector(can[1])
    c3 = one_hot_vector(can[2])

    image_can = np.array([c1, c2, c3])
    return image_can


# test this code to check working well
if __name__ == "__main__":
    """load normal_run_data.txt"""
    can_feature = ["Timestamp", "blank", "ID", "zero", "DLC", "Data"]
    data_for_test = pd.read_csv(
        "dataset/Training set/normal_run_data.txt",
        sep="    ",
        engine="python",
        encoding="cp949",
        header=None,
        names=can_feature,
    )
    df = data_for_test["ID"]
    df = df.drop(df.index[988871])  # drop None index

    for i in range(len(df)):  # extract needed part
        df.at[i] = df.at[i][5:8]
    print(df.head())

''' I have to add a function to make can images '''
