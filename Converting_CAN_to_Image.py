import numpy as np
import pandas as pd
from PIL import Image


def one_hot_vector(c):
    # convert element of can_data to one hot vector to select can image position
    # can element is white(255) and else is black(0)
    #    can order: 1  2  3  4  5  6  7  8  9  0  a  b  c  d  e  f
    ohv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    if c == "1":
        ohv[0] = 255
    elif c == "2":
        ohv[1] = 255
    elif c == "3":
        ohv[2] = 255
    elif c == "4":
        ohv[3] = 255
    elif c == "5":
        ohv[4] = 255
    elif c == "6":
        ohv[5] = 255
    elif c == "7":
        ohv[6] = 255
    elif c == "8":
        ohv[7] = 255
    elif c == "9":
        ohv[8] = 255
    elif c == "0":
        ohv[9] = 255
    elif c == "a":
        ohv[10] = 255
    elif c == "b":
        ohv[11] = 255
    elif c == "c":
        ohv[12] = 255
    elif c == "d":
        ohv[13] = 255
    elif c == "e":
        ohv[14] = 255
    elif c == "f":
        ohv[15] = 255
    return ohv


def convert_to_can_data_to_one_hot_vector(can):
    # convert can_data to image

    c1 = one_hot_vector(can[0])
    c2 = one_hot_vector(can[1])
    c3 = one_hot_vector(can[2])

    image_can = np.array([c1, c2, c3])
    return image_can


def make_can_image(data):
    can_image = np.zeros((1, 16))
    # set size of image to 16 X ( 3 * 64 )
    # colum is Hexadecimal presentation from CAN ID value
    # row is  3 (CAN ID length) * 64 (optimal row size in reference paper)
    num = 1
    for w in range(0, len(data),64):                                   # <== I have to check this part
        lst = data.at[w:w+64]
        for i in range(0, len(lst)):
            add_can = convert_to_can_data_to_one_hot_vector(lst.iloc[i][0])
            can_image = np.vstack([can_image, add_can])
            can_image = np.delete(can_image, (0), axis=0)
            int_can_image = can_image.astype(int)

        image = Image.fromarray(make_can_image(df))
        image = image.convert("RGB")
        image.save('can_img_{}.png'.format(num))
        num+=1
    return int_can_image


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

""" I have to add a function to make can images """
