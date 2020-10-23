import numpy as np
import pandas as pd
import argparse
from scipy.spatial import distance as dist
"""
fdgdfgdf
TO DO
remove duplicated code
"""

def getLandmark2D(data):
    """
        Input: csv landmarks file, parses for relevant landmarks from 2D  
        Landmarks (pixels): LOCP, ROCP, LICP, RICP NB ROCE RICE LOCE LICE.
        Returns: dataframe composed of relevant landmarks for eye.


    """
    df2D = pd.DataFrame(data, index=pd.Index(['x','y'],name='Axis'),
                        columns=pd.Index(['LOCP','ROCP','LICP','RICP',
                                        'NB','ROCE','RICE','LOCE','LICE'],
                                        name='ROI'))

    # landmarks based on reflected image. 
    ############
    # TO DO
    """
        strip white spaces in name
        eliminate hard coding
        try different combos to get more accurate PD
    """
    ############
    #LOCP
    df2D['LOCP']['x'] = data[' eye_lmk_x_51'] # left outer corner pupil x 
    df2D['LOCP']['y'] = data[' eye_lmk_y_51'] # left outer corner pupil y
    #ROCP
    df2D['ROCP']['x'] = data[' eye_lmk_x_27'] # right outer corner pupil x
    df2D['ROCP']['y'] = data[' eye_lmk_y_27'] # right outer corner pupil y
    #LICP
    df2D['LICP']['x'] = data[' eye_lmk_x_55'] # left inner corner pupil x
    df2D['LICP']['y'] = data[' eye_lmk_y_55'] # left inner corner pupil y
    #RICP
    df2D['RICP']['x'] = data[' eye_lmk_x_23'] # right inner corner pupil x
    df2D['RICP']['y'] = data[' eye_lmk_y_23'] # right inner corner pupil y
    #NB
    df2D['NB']['x'] = data[' x_27'] # nose bridge x
    df2D['NB']['y'] = data[' y_27'] # nose bridge y
    #ROCE
    df2D['ROCE']['x'] = data[' x_36'] # right outer corner eye x
    df2D['ROCE']['y'] = data[' y_36'] # right outer corner eye y
    #RICE
    df2D['RICE']['x'] = data[' x_39'] # right inner corner eye x
    df2D['RICE']['y'] = data[' y_39'] # right inner corner eye y
    #LOCE
    df2D['LOCE']['x'] = data[' x_45'] # left outer corner eye x
    df2D['LOCE']['y'] = data[' y_45'] # left outer corner eye y
    #LICE
    df2D['LICE']['x'] = data[' x_42'] # left inner corner eye x
    df2D['LICE']['y'] = data[' y_42'] # left inner corner eye y

    return df2D


def getLandmark3D(data):
    """
        Input: csv landmarks file, parses for relevant landmarks from 3D  
        Landmarks (mm): LOCP, ROCP, LICP, RICP NB ROCE RICE LOCE LICE.
        Returns: dataframe composed of relevant landmarks for eye.


    """
    df3D = pd.DataFrame(data, index=pd.Index(['X','Y','Z'],name='Axis'),columns=pd.Index(['LOCP','ROCP','LICP','RICP','NB','ROCE','RICE','LOCE','LICE'],name='ROI'))
    df2D = df3D
    # landmarks based on reflected image. 
    ############
    # TO DO
    """
        strip white spaces in name
        eliminate hard coding
        try different combos to get more accurate PD
    """
    ############
    #LOCP
    df3D['LOCP']['X'] = data[' eye_lmk_X_51'] # left outer corner pupil X 
    df3D['LOCP']['Y'] = data[' eye_lmk_Y_51'] # left outer corner pupil Y
    df3D['LOCP']['Z'] = data[' eye_lmk_Z_51'] # left outer corner pupil Z
    #ROCP
    df3D['ROCP']['X'] = data[' eye_lmk_X_27'] # right outer corner pupil X
    df3D['ROCP']['Y'] = data[' eye_lmk_Y_27'] # right outer corner pupil Y
    df3D['ROCP']['Z'] = data[' eye_lmk_Z_27'] # right outer corner pupil Z
    #LICP
    df3D['LICP']['X'] = data[' eye_lmk_X_55'] # left inner corner pupil X
    df3D['LICP']['Y'] = data[' eye_lmk_Y_55'] # left inner corner pupil Y
    df3D['LICP']['Z'] = data[' eye_lmk_Z_55'] # left inner corner pupil Z
    #RICP
    df3D['RICP']['X'] = data[' eye_lmk_X_23'] # right inner corner pupil X
    df3D['RICP']['Y'] = data[' eye_lmk_Y_23'] # right inner corner pupil Y
    df3D['RICP']['Z'] = data[' eye_lmk_Z_23'] # right inner corner pupil Z
    #NB
    df3D['NB']['X'] = data[' X_27'] # nose bridge X
    df3D['NB']['Y'] = data[' Y_27'] # nose bridge Y
    df3D['NB']['Z'] = data[' Z_27'] # nose bridge Z
    #ROCE
    df3D['ROCE']['X'] = data[' X_36'] # right outer corner eye X
    df3D['ROCE']['Y'] = data[' Y_36'] # right outer corner eye Y
    df3D['ROCE']['Z'] = data[' Z_36'] # right outer corner eye Z
    #RICE
    df3D['RICE']['X'] = data[' X_39'] # right inner corner eye X
    df3D['RICE']['Y'] = data[' Y_39'] # right inner corner eye Y
    df3D['RICE']['Z'] = data[' Z_39'] # right inner corner eye Z
    #LOCE
    df3D['LOCE']['X'] = data[' X_45'] # left outer corner eye X
    df3D['LOCE']['Y'] = data[' Y_45'] # left outer corner eye Y
    df3D['LOCE']['Z'] = data[' Z_45'] # left outer corner eye Z
    #LICE
    df3D['LICE']['X'] = data[' X_42'] # left inner corner eye X
    df3D['LICE']['Y'] = data[' Y_42'] # left inner corner eye Y
    df3D['LICE']['Z'] = data[' Z_42'] # left inner corner eye Z
    return df3D

    
def calcPD(data):
    """
        Input: eye landmark data frame.
        performs calculations on different landmarks to get PD
        Returns: list of PD values. Mono and Dual PD  
    """
    df = getLandmark2D(data)
    print('start here')
    print(df['LICP'])
    print(df['RICP'])
    euclidDist = dist.euclidean(df['LICP'],df['RICP'])
    # ((df['LICP'] - df['RICP'])**2).sum()
    print(euclidDist)
    print("all ret values printed in calcPD")

    df3d = getLandmark3D
    return [df['LICP'],df['RICP'],euclidDist]
    # print(np.sqrt(euclidDist))
    #calculating 3d distance
    # ocp_pd = 
    #est the mono pd by finding avg in iris

    # print(eyeX39)
    # avgeye = (eyeX39 - eyeX36)/2

    # print(data)
    # print(key_lmk_X27,key_lmk_Y27,key_lmk_Z27,key_lmk_X51,key_lmk_Y51,key_lmk_Z51)
    # pdX = key_lmk_X51 - key_lmk_X27
    # pdY = key_lmk_Y51 - key_lmk_Y27
    # pdZ = key_lmk_Z51 - key_lmk_Z27
    # monoPDX = eyeX36 - noseX27
    # print("here is my PD est for X {}".format(pdX))
    # print("here is my PD est for Y {}".format(pdY))
    # print("here is my PD est for Z {}".format(pdZ))
    # print("here is mono PD est for X {}".format(monoPDX))
    # print("here is eye avg est for X {}".format(avgeye))
    # print("calculated  mono pd est {}".format(avgeye-noseX27 ))

