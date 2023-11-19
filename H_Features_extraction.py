import numpy as np
import pandas as pd
import glob
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage import io, exposure, morphology, color, feature, img_as_ubyte
import mahotas as mh
import os 

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


SIZE_X = 224 #Resize images (height  = X, width = Y)
SIZE_Y = 224

#   _______ _____            _____ _   _ 
#  |__   __|  __ \     /\   |_   _| \ | |
#     | |  | |__) |   /  \    | | |  \| |
#     | |  |  _  /   / /\ \   | | | . ` |
#     | |  | | \ \  / ____ \ _| |_| |\  |
#     |_|  |_|  \_\/_/    \_\_____|_| \_|


#Read isic2019 data
path = 'D:/CS ML/dataset/ISIC-2017_Training_Data/'
names = os.listdir(path)


filenames = []
C_r=[]
C_g=[]
C_b=[]
correlation=[]
dissimilarity = []
homogeneity=[]
energy=[]
contrast = []
values = []

counter = 0

filenames = []   
for name in names:

    # if counter == 30:
    #     break
    # counter += 1
    
    # For isic2018 or isic2019
    filenames.append(os.path.splitext(name)[0])
    print(name)
    
    #Dull razor algorithm to remove hair and noise from images
    img = cv2.imread(path+name, cv2.IMREAD_COLOR)
    structuring_element = morphology.disk(3)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    imgtmp1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #applying a blackhat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
    imgtmp2 = cv2.morphologyEx(imgtmp1, cv2.MORPH_BLACKHAT, kernel)
    #0=skin and 255=hair
    lowbound=15
    ret, mask = cv2.threshold(imgtmp2, lowbound, 255, cv2.THRESH_BINARY)
    #inpainting
    inpaintmat=3
    img_final = cv2.inpaint(img, mask, inpaintmat ,cv2.INPAINT_TELEA)
    

    # # Extracting features
    gray_img = color.rgb2gray(img)
    
    # 1] Color variegation:
    lesion_r = img[:, :, 0]
    lesion_g = img[:, :, 1]
    lesion_b = img[:, :, 2]
    
    C_r.append(np.std(lesion_r) / np.max(lesion_r))
    C_g.append(np.std(lesion_g) / np.max(lesion_g))
    C_b.append(np.std(lesion_b) / np.max(lesion_b))
    
    
    # 2] Texture
    glcm = feature.graycomatrix(image=img_as_ubyte(gray_img), distances=[1],
                                angles=[0, np.pi/4, np.pi/2, np.pi * 3/2],
                                symmetric=True, normed=True)

    correlation.append(np.mean(feature.graycoprops(glcm, prop='correlation')))
    dissimilarity.append(np.mean(feature.graycoprops(glcm, prop='dissimilarity')))
    homogeneity.append(np.mean(feature.graycoprops(glcm, prop='homogeneity')))
    energy.append(np.mean(feature.graycoprops(glcm, prop='energy')))
    contrast.append(np.mean(feature.graycoprops(glcm, prop='contrast')))


    # 3] Threshold analysis statistics
    value = mh.features.tas(gray_img)
    values.append(value.transpose())
    

val = np.array(values)


df_metadata = pd.read_csv('D:\CS ML\Data files\ISIC-2017_Training_Part3_GroundTruth.csv')
l = df_metadata[['label']]

features = ['Image','label','StdR', 'StdG', 'StdB', 'Correlation', 'Dissimilarity', 'Homogeneity', 'Energy', 'Contrast']
feature_df = pd.DataFrame([filenames, l['label'].tolist(), C_r, C_g, C_b, correlation, dissimilarity, homogeneity, energy, contrast])
feature_df = feature_df.transpose()
feature_df.columns = features


#save as csv
feature_df.to_csv("D:/CS ML/Data files/ISIC_2017_all_ExtractedFeatures.csv", index = False)

############################################################################################################################################################################################

#This file preprocesses metadata for train, val and test folders separately or preprocess the entire metadata file and further split into train, test and val

import cv2
import numpy as np
import os
import math

# Get the training classes names and store them in a list
# Here we use folder names for class names
train_path = 'D:/CS ML/dataset'

training_names = os.listdir(train_path)

# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
file_names = []
image_classes = []
class_id = 0

#To make it easy to list all file names in a directory let us define a function
def imglist(path):    
    return [os.path.join(path, f) for f in os.listdir(path)]

#Fill the placeholder empty lists with image path, classes, and add class ID number
    
for training_name in training_names:
    #print(training_name)
    direct = os.path.join(train_path, training_name)
    print(direct)
    class_path = imglist(direct)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1
    
for image_path in image_paths:
    head, tail = os.path.split(image_path)
    file_name, file_extension = os.path.splitext(tail)
    # print(file_name)
    file_names+=[file_name]
    

import pandas as pd
df = pd.DataFrame([file_names,image_paths,image_classes]) #Each list would be added as a row
df = df.transpose() #To Transpose and make each rows as columns
df.columns=['Image','Path','Label'] #Rename the columns

df.to_csv("D:\CS ML\Data files\ISIC2017_label.csv", index = False)


#Add Metadata to all image names
df_metadata = pd.read_csv('D:/CS ML/Data files/ISIC-2017_Training_Data_metadata.csv')

#Combine both dataframes as [image, path, label, metadata]
dfinal = df.merge(df_metadata, how='left', left_on='Image', right_on='image_id')

dfinal.drop('image_id', axis=1, inplace=True)
#Rearrange and place label at the end
dfinal = dfinal[["Image", "Path", "Label", "age_approximate", "sex"]]


# print the percentage of values occuring in a specific metadata column
def perc_data(column, df):
    a = column.unique()
    x = column.value_counts()
    rows = df.shape[0]
    print("CLASS\t\t : \tPERCENTAGE")
    print("------------------------------------")
    for i in a:
        print(f"{i}\t\t : \t{(x[i]/rows)*100}")
    print(f"NULL\t\t : \t{(column.isna().sum()/rows)*100}")
perc_data(dfinal.age_approximate, dfinal)
perc_data(dfinal.sex, dfinal)
dfinal.shape


# Replace all empty spaces and zeroes in age column with the mean age
print(dfinal.age_approximate.mean())
print(dfinal.age_approximate.isna().sum())
dfinal["age_approximate"] = dfinal["age_approximate"].fillna(0)
dfinal["age_approximate"] = dfinal["age_approximate"].replace(0, dfinal.age_approximate.mean())

#Since male occurs with max freq, unknown values are replaced with male
dfinal["sex"].replace({"unknown": "male"}, inplace=True)
print(dfinal.sex.isna().sum())
dfinal["sex"] = dfinal["sex"].fillna(0)
dfinal["sex"] = dfinal["sex"].replace(0, "male")

perc_data(dfinal.age_approximate, dfinal)
perc_data(dfinal.sex, dfinal)


#PREPROCESS DATA
#One hot encode sex
one_hot = pd.get_dummies(dfinal['sex'])
# Drop column sex as it is now encoded
dfinal = dfinal.drop('sex',axis = 1)
# Join the encoded df
dfinal = dfinal.join(one_hot)

dfinal.drop('Path', axis=1, inplace=True)


dfinal.to_csv("D:/CS ML/Data files/ISIC2017_train_imname_metadata_label.csv", index = False)


############################################################################################################################################################################################


#Produce Dimensionality reduced metadata+feature file for train val and test

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#Read Metadata of train, val and test separately
df_metadata = pd.read_csv('D:\CS ML\ISIC2017_train_imname_metadata_label.csv')
df_features = pd.read_csv('D:\CS ML\Data files\ISIC_2017_all_ExtractedFeatures.csv')

img = []
img = df_metadata['Image']

#Combine both dataframes as [image, path, label, metadata]
dfinal = df_metadata.merge(df_features, how='left', left_on='Image', right_on='Image')

# FIT & APPLY LDA TRANSFORM
data = dfinal[['age_approximate', 'female', 'male', 'StdR', 'StdG', 'StdB', 'Correlation', 'Dissimilarity', 'Homogeneity', 'Energy', 'Contrast']]
l = dfinal[['label']]

#Normalize to the range [0,1] after LDA
scalar = MinMaxScaler()
# data_lda = scalar.fit_transform(data_lda_)
data_scalar = scalar.fit_transform(data)

df_lda = pd.DataFrame(data_scalar)
df_lda.columns = ['age_approximate', 'female', 'male', 'StdR', 'StdG', 'StdB', 'Correlation', 'Dissimilarity', 'Homogeneity', 'Energy', 'Contrast']
df_lda['Image'] = img


data_final = dfinal[['Image', 'label']]
final = data_final.merge(df_lda, how='left', left_on='Image', right_on='Image')

#save as csv
final.to_csv("D:/CS ML/ISIC2017_train_metadata_all.csv", index = False)

#  __      __     _      _____ _____       _______ _____ ____  _   _ 
#  \ \    / /\   | |    |_   _|  __ \   /\|__   __|_   _/ __ \| \ | |
#   \ \  / /  \  | |      | | | |  | | /  \  | |    | || |  | |  \| |
#    \ \/ / /\ \ | |      | | | |  | |/ /\ \ | |    | || |  | | . ` |
#     \  / ____ \| |____ _| |_| |__| / ____ \| |   _| || |__| | |\  |
#      \/_/    \_\______|_____|_____/_/    \_\_|  |_____\____/|_| \_|

#Read isic2019 data
path = 'D:/CS ML/dataset/ISIC-2017_Validation_Data/'
names = os.listdir(path)


filenames = []
C_r=[]
C_g=[]
C_b=[]
correlation=[]
dissimilarity = []
homogeneity=[]
energy=[]
contrast = []
values = []

counter = 0

filenames = []   
for name in names:

    # if counter == 30:
    #     break
    # counter += 1

    filenames.append(os.path.splitext(name)[0])
    print(name)
    
    #Dull razor algorithm to remove hair and noise from images
    img = cv2.imread(path+name, cv2.IMREAD_COLOR)
    structuring_element = morphology.disk(3)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    imgtmp1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #applying a blackhat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
    imgtmp2 = cv2.morphologyEx(imgtmp1, cv2.MORPH_BLACKHAT, kernel)
    #0=skin and 255=hair
    lowbound=15
    ret, mask = cv2.threshold(imgtmp2, lowbound, 255, cv2.THRESH_BINARY)
    #inpainting
    inpaintmat=3
    img_final = cv2.inpaint(img, mask, inpaintmat ,cv2.INPAINT_TELEA)
    

    # # Extracting features
    gray_img = color.rgb2gray(img)
    
    # 1] Color variegation:
    lesion_r = img[:, :, 0]
    lesion_g = img[:, :, 1]
    lesion_b = img[:, :, 2]
    
    C_r.append(np.std(lesion_r) / np.max(lesion_r))
    C_g.append(np.std(lesion_g) / np.max(lesion_g))
    C_b.append(np.std(lesion_b) / np.max(lesion_b))
    
    
    # 2] Texture
    glcm = feature.graycomatrix(image=img_as_ubyte(gray_img), distances=[1],
                                angles=[0, np.pi/4, np.pi/2, np.pi * 3/2],
                                symmetric=True, normed=True)

    correlation.append(np.mean(feature.graycoprops(glcm, prop='correlation')))
    dissimilarity.append(np.mean(feature.graycoprops(glcm, prop='dissimilarity')))
    homogeneity.append(np.mean(feature.graycoprops(glcm, prop='homogeneity')))
    energy.append(np.mean(feature.graycoprops(glcm, prop='energy')))
    contrast.append(np.mean(feature.graycoprops(glcm, prop='contrast')))


    # 3] Threshold analysis statistics
    value = mh.features.tas(gray_img)
    values.append(value.transpose())
    

val = np.array(values)


df_metadata = pd.read_csv('D:\CS ML\Data files\ISIC-2017_Validation_Part3_GroundTruth.csv')
l = df_metadata[['label']]

features = ['Image','label','StdR', 'StdG', 'StdB', 'Correlation', 'Dissimilarity', 'Homogeneity', 'Energy', 'Contrast']
feature_df = pd.DataFrame([filenames, l['label'].tolist(), C_r, C_g, C_b, correlation, dissimilarity, homogeneity, energy, contrast])
feature_df = feature_df.transpose()
feature_df.columns = features


#save as csv
feature_df.to_csv("D:/CS ML/Data files/ISIC_2017_all_ExtractedFeatures_val.csv", index = False)

############################################################################################################################################################################################

#This file preprocesses metadata for train, val and test folders separately or preprocess the entire metadata file and further split into train, test and val

import cv2
import numpy as np
import os
import math

# Get the training classes names and store them in a list
# Here we use folder names for class names
train_path = 'D:/CS ML/dataset'

training_names = os.listdir(train_path)

# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
file_names = []
image_classes = []
class_id = 0

#To make it easy to list all file names in a directory let us define a function
def imglist(path):    
    return [os.path.join(path, f) for f in os.listdir(path)]

#Fill the placeholder empty lists with image path, classes, and add class ID number
    
for training_name in training_names:
    #print(training_name)
    direct = os.path.join(train_path, training_name)
    print(direct)
    class_path = imglist(direct)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1
    
for image_path in image_paths:
    head, tail = os.path.split(image_path)
    file_name, file_extension = os.path.splitext(tail)
    # print(file_name)
    file_names+=[file_name]
    

import pandas as pd
df = pd.DataFrame([file_names,image_paths,image_classes]) #Each list would be added as a row
df = df.transpose() #To Transpose and make each rows as columns
df.columns=['Image','Path','Label'] #Rename the columns

df.to_csv("D:\CS ML\Data files\ISIC2017_label_val.csv", index = False)


#Add Metadata to all image names
df_metadata = pd.read_csv('Data files/ISIC-2017_Validation_Data_metadata.csv')

#Combine both dataframes as [image, path, label, metadata]
dfinal = df.merge(df_metadata, how='left', left_on='Image', right_on='image_id')

dfinal.drop('image_id', axis=1, inplace=True)
#Rearrange and place label at the end
dfinal = dfinal[["Image", "Path", "Label", "age_approximate", "sex"]]


# print the percentage of values occuring in a specific metadata column
def perc_data(column, df):
    a = column.unique()
    x = column.value_counts()
    rows = df.shape[0]
    print("CLASS\t\t : \tPERCENTAGE")
    print("------------------------------------")
    for i in a:
        print(f"{i}\t\t : \t{(x[i]/rows)*100}")
    print(f"NULL\t\t : \t{(column.isna().sum()/rows)*100}")
perc_data(dfinal.age_approximate, dfinal)
perc_data(dfinal.sex, dfinal)
dfinal.shape


# Replace all empty spaces and zeroes in age column with the mean age
print(dfinal.age_approximate.mean())
print(dfinal.age_approximate.isna().sum())
dfinal["age_approximate"] = dfinal["age_approximate"].fillna(0)
dfinal["age_approximate"] = dfinal["age_approximate"].replace(0, dfinal.age_approximate.mean())

#Since male occurs with max freq, unknown values are replaced with male
dfinal["sex"].replace({"unknown": "male"}, inplace=True)
print(dfinal.sex.isna().sum())
dfinal["sex"] = dfinal["sex"].fillna(0)
dfinal["sex"] = dfinal["sex"].replace(0, "male")

perc_data(dfinal.age_approximate, dfinal)
perc_data(dfinal.sex, dfinal)


#PREPROCESS DATA
#One hot encode sex
one_hot = pd.get_dummies(dfinal['sex'])
# Drop column sex as it is now encoded
dfinal = dfinal.drop('sex',axis = 1)
# Join the encoded df
dfinal = dfinal.join(one_hot)

dfinal.drop('Path', axis=1, inplace=True)


dfinal.to_csv("D:/CS ML/Data files/ISIC2017_val_imname_metadata_label.csv", index = False)


############################################################################################################################################################################################


#Produce Dimensionality reduced metadata+feature file for train val and test

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#Read Metadata of train, val and test separately
df_metadata = pd.read_csv('D:/CS ML/Data files/ISIC2017_val_imname_metadata_label.csv')
df_features = pd.read_csv('D:\CS ML\Data files\ISIC_2017_all_ExtractedFeatures_val.csv')

img = []
img = df_metadata['Image']

#Combine both dataframes as [image, path, label, metadata]
dfinal = df_metadata.merge(df_features, how='left', left_on='Image', right_on='Image')

# FIT & APPLY LDA TRANSFORM
data = dfinal[['age_approximate', 'female', 'male', 'StdR', 'StdG', 'StdB', 'Correlation', 'Dissimilarity', 'Homogeneity', 'Energy', 'Contrast']]
l = dfinal[['label']]

#Normalize to the range [0,1] after LDA
scalar = MinMaxScaler()
# data_lda = scalar.fit_transform(data_lda_)
data_scalar = scalar.fit_transform(data)

df_lda = pd.DataFrame(data_scalar)
df_lda.columns = ['age_approximate', 'female', 'male', 'StdR', 'StdG', 'StdB', 'Correlation', 'Dissimilarity', 'Homogeneity', 'Energy', 'Contrast']
df_lda['Image'] = img


data_final = dfinal[['Image', 'label']]
final = data_final.merge(df_lda, how='left', left_on='Image', right_on='Image')

#save as csv
final.to_csv("D:/CS ML/ISIC2017_val_metadata_all.csv", index = False)






#   _______ ______  _____ _______ 
#  |__   __|  ____|/ ____|__   __|
#     | |  | |__  | (___    | |   
#     | |  |  __|  \___ \   | |   
#     | |  | |____ ____) |  | |   
#     |_|  |______|_____/   |_| 

#Read isic2019 data
path = 'D:/CS ML/dataset/ISIC-2017_Test_v2_Data/'
names = os.listdir(path)


filenames = []
C_r=[]
C_g=[]
C_b=[]
correlation=[]
dissimilarity = []
homogeneity=[]
energy=[]
contrast = []
values = []

counter = 0

filenames = []   
for name in names:

    # if counter == 30:
    #     break
    # counter += 1

    filenames.append(os.path.splitext(name)[0])
    print(name)
    
    #Dull razor algorithm to remove hair and noise from images
    img = cv2.imread(path+name, cv2.IMREAD_COLOR)
    structuring_element = morphology.disk(3)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    imgtmp1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #applying a blackhat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
    imgtmp2 = cv2.morphologyEx(imgtmp1, cv2.MORPH_BLACKHAT, kernel)
    #0=skin and 255=hair
    lowbound=15
    ret, mask = cv2.threshold(imgtmp2, lowbound, 255, cv2.THRESH_BINARY)
    #inpainting
    inpaintmat=3
    img_final = cv2.inpaint(img, mask, inpaintmat ,cv2.INPAINT_TELEA)
    

    # # Extracting features
    gray_img = color.rgb2gray(img)
    
    # 1] Color variegation:
    lesion_r = img[:, :, 0]
    lesion_g = img[:, :, 1]
    lesion_b = img[:, :, 2]
    
    C_r.append(np.std(lesion_r) / np.max(lesion_r))
    C_g.append(np.std(lesion_g) / np.max(lesion_g))
    C_b.append(np.std(lesion_b) / np.max(lesion_b))
    
    
    # 2] Texture
    glcm = feature.graycomatrix(image=img_as_ubyte(gray_img), distances=[1],
                                angles=[0, np.pi/4, np.pi/2, np.pi * 3/2],
                                symmetric=True, normed=True)

    correlation.append(np.mean(feature.graycoprops(glcm, prop='correlation')))
    dissimilarity.append(np.mean(feature.graycoprops(glcm, prop='dissimilarity')))
    homogeneity.append(np.mean(feature.graycoprops(glcm, prop='homogeneity')))
    energy.append(np.mean(feature.graycoprops(glcm, prop='energy')))
    contrast.append(np.mean(feature.graycoprops(glcm, prop='contrast')))


    # 3] Threshold analysis statistics
    value = mh.features.tas(gray_img)
    values.append(value.transpose())
    

val = np.array(values)


df_metadata = pd.read_csv('D:\CS ML\Data files\ISIC-2017_Test_v2_Part3_GroundTruth.csv')
l = df_metadata[['label']]

features = ['Image','label','StdR', 'StdG', 'StdB', 'Correlation', 'Dissimilarity', 'Homogeneity', 'Energy', 'Contrast']
feature_df = pd.DataFrame([filenames, l['label'].tolist(), C_r, C_g, C_b, correlation, dissimilarity, homogeneity, energy, contrast])
feature_df = feature_df.transpose()
feature_df.columns = features


#save as csv
feature_df.to_csv("D:/CS ML/Data files/ISIC_2017_all_ExtractedFeatures_test.csv", index = False)

############################################################################################################################################################################################

#This file preprocesses metadata for train, val and test folders separately or preprocess the entire metadata file and further split into train, test and val

import cv2
import numpy as np
import os
import math

# Get the training classes names and store them in a list
# Here we use folder names for class names
train_path = 'D:/CS ML/dataset'

training_names = os.listdir(train_path)

# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
file_names = []
image_classes = []
class_id = 0

#To make it easy to list all file names in a directory let us define a function
def imglist(path):    
    return [os.path.join(path, f) for f in os.listdir(path)]

#Fill the placeholder empty lists with image path, classes, and add class ID number
    
for training_name in training_names:
    #print(training_name)
    direct = os.path.join(train_path, training_name)
    print(direct)
    class_path = imglist(direct)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1
    
for image_path in image_paths:
    head, tail = os.path.split(image_path)
    file_name, file_extension = os.path.splitext(tail)
    # print(file_name)
    file_names+=[file_name]
    

import pandas as pd
df = pd.DataFrame([file_names,image_paths,image_classes]) #Each list would be added as a row
df = df.transpose() #To Transpose and make each rows as columns
df.columns=['Image','Path','Label'] #Rename the columns

df.to_csv("D:\CS ML\Data files\ISIC2017_label_test.csv", index = False)


#Add Metadata to all image names
df_metadata = pd.read_csv('D:\CS ML\Data files\ISIC-2017_Test_v2_Data_metadata.csv')

#Combine both dataframes as [image, path, label, metadata]
dfinal = df.merge(df_metadata, how='left', left_on='Image', right_on='image_id')

dfinal.drop('image_id', axis=1, inplace=True)
#Rearrange and place label at the end
dfinal = dfinal[["Image", "Path", "Label", "age_approximate", "sex"]]


# print the percentage of values occuring in a specific metadata column
def perc_data(column, df):
    a = column.unique()
    x = column.value_counts()
    rows = df.shape[0]
    print("CLASS\t\t : \tPERCENTAGE")
    print("------------------------------------")
    for i in a:
        print(f"{i}\t\t : \t{(x[i]/rows)*100}")
    print(f"NULL\t\t : \t{(column.isna().sum()/rows)*100}")
perc_data(dfinal.age_approximate, dfinal)
perc_data(dfinal.sex, dfinal)
dfinal.shape


# Replace all empty spaces and zeroes in age column with the mean age
print(dfinal.age_approximate.mean())
print(dfinal.age_approximate.isna().sum())
dfinal["age_approximate"] = dfinal["age_approximate"].fillna(0)
dfinal["age_approximate"] = dfinal["age_approximate"].replace(0, dfinal.age_approximate.mean())

#Since male occurs with max freq, unknown values are replaced with male
dfinal["sex"].replace({"unknown": "male"}, inplace=True)
print(dfinal.sex.isna().sum())
dfinal["sex"] = dfinal["sex"].fillna(0)
dfinal["sex"] = dfinal["sex"].replace(0, "male")

perc_data(dfinal.age_approximate, dfinal)
perc_data(dfinal.sex, dfinal)


#PREPROCESS DATA
#One hot encode sex
one_hot = pd.get_dummies(dfinal['sex'])
# Drop column sex as it is now encoded
dfinal = dfinal.drop('sex',axis = 1)
# Join the encoded df
dfinal = dfinal.join(one_hot)

dfinal.drop('Path', axis=1, inplace=True)


dfinal.to_csv("D:/CS ML/Data files/ISIC2017_test_imname_metadata_label.csv", index = False)


############################################################################################################################################################################################


#Produce Dimensionality reduced metadata+feature file for train val and test

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#Read Metadata of train, val and test separately
df_metadata = pd.read_csv('D:/CS ML/Data files/ISIC2017_test_imname_metadata_label.csv')
df_features = pd.read_csv('D:\CS ML\Data files\ISIC_2017_all_ExtractedFeatures_test.csv')

img = []
img = df_metadata['Image']

#Combine both dataframes as [image, path, label, metadata]
dfinal = df_metadata.merge(df_features, how='left', left_on='Image', right_on='Image')

# FIT & APPLY LDA TRANSFORM
data = dfinal[['age_approximate', 'female', 'male', 'StdR', 'StdG', 'StdB', 'Correlation', 'Dissimilarity', 'Homogeneity', 'Energy', 'Contrast']]
l = dfinal[['label']]

#Normalize to the range [0,1] after LDA
scalar = MinMaxScaler()
# data_lda = scalar.fit_transform(data_lda_)
data_scalar = scalar.fit_transform(data)

df_lda = pd.DataFrame(data_scalar)
df_lda.columns = ['age_approximate', 'female', 'male', 'StdR', 'StdG', 'StdB', 'Correlation', 'Dissimilarity', 'Homogeneity', 'Energy', 'Contrast']
df_lda['Image'] = img


data_final = dfinal[['Image', 'label']]
final = data_final.merge(df_lda, how='left', left_on='Image', right_on='Image')

#save as csv
final.to_csv("D:/CS ML/ISIC2017_test_metadata_all.csv", index = False)