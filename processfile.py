import numpy as np
import pandas as pd
import argparse

#rewrite as func
#  # 
#     data = pd.read_csv('andrewcalib.csv')
#     nodata = pd.read_csv('djfacenocalib.csv')
def getLandmark(data):
    df = pd.DataFrame(data, index=pd.Index(['X','Y','Z'],name='Axis'),columns=pd.Index(['LOCP','ROCP','LICP','RICP','NB','ROCE','RICE','LOCE','LICE'],name='ROI'))
    # landmarks based on reflected image. 
    #LOCP
    df['LOCP']['X'] = data[' eye_lmk_X_51'] # left outer corner pupil X 
    df['LOCP']['Y'] = data[' eye_lmk_Y_51'] # left outer corner pupil Y
    df['LOCP']['Z'] = data[' eye_lmk_Z_51'] # left outer corner pupil Z
    #ROCP
    df['ROCP']['X'] = data[' eye_lmk_X_27'] # right outer corner pupil X
    df['ROCP']['Y'] = data[' eye_lmk_Y_27'] # right outer corner pupil Y
    df['ROCP']['Z'] = data[' eye_lmk_Z_27'] # right outer corner pupil Z
    #LICP
    df['LICP']['X'] = data[' eye_lmk_X_55'] # left inner corner pupil X
    df['LICP']['Y'] = data[' eye_lmk_Y_55'] # left inner corner pupil Y
    df['LICP']['Z'] = data[' eye_lmk_Z_55'] # left inner corner pupil Z
    #RICP
    df['RICP']['X'] = data[' eye_lmk_X_23'] # right inner corner pupil X
    df['RICP']['Y'] = data[' eye_lmk_Y_23'] # right inner corner pupil Y
    df['RICP']['Z'] = data[' eye_lmk_Z_23'] # right inner corner pupil Z
    #NB
    df['NB']['X'] = data[' X_27'] # nose bridge X
    df['NB']['Y'] = data[' Y_27'] # nose bridge Y
    df['NB']['Z'] = data[' Z_27'] # nose bridge Z
    #ROCE
    df['ROCE']['X'] = data[' X_36'] # right outer corner eye X
    df['ROCE']['Y'] = data[' Y_36'] # right outer corner eye Y
    df['ROCE']['Z'] = data[' Z_36'] # right outer corner eye Z
    #RICE
    df['RICE']['X'] = data[' X_39'] # right inner corner eye X
    df['RICE']['Y'] = data[' Y_39'] # right inner corner eye Y
    df['RICE']['Z'] = data[' Z_39'] # right inner corner eye Z
    #LOCE
    df['LOCE']['X'] = data[' X_45'] # left outer corner eye X
    df['LOCE']['Y'] = data[' Y_45'] # left outer corner eye Y
    df['LOCE']['Z'] = data[' Z_45'] # left outer corner eye Z
    #LICE
    df['LICE']['X'] = data[' X_42'] # left inner corner eye X
    df['LICE']['Y'] = data[' Y_42'] # left inner corner eye Y
    df['LICE']['Z'] = data[' Z_42'] # left inner corner eye Z
    return df

    
def calcPD(data):
    df = getLandmark(data)
    print('start here')
    print(df['LICP'])
    print(df['RICP'])
    geom = ((df['LICP'] - df['RICP'])**2).sum()
    print(geom)
    print(np.sqrt(geom))
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

data = pd.read_csv('andrewFace.csv')
# data = pd.read_csv('andrewcalib.csv')

calcPD(data)
