import os
import shutil
from PIL import Image
import numpy as np

def one_hot(x) :
    a = np.zeros(20, dtype = "float32")
    a[x] = 1.0
    return a
def load_data_from_cityscapes(picture_path, label_path):
  catcolor = [(0,0,0),(111,74,0),(81,0,81),(128, 64,128),(244, 35,232),(250,170,160),(230,150,140),(70,70,70),
  (102,102,156),(190,153,153),(180,165,180),(150,100,100),(150,120,90),(153,153,153),(250,170,30),(220,220,0),
  (107,142,35),(152,251,152),(70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),(0,0,90),(0,0,110),
  (0,80,100),(0,0,230),(119,11,32),(0,0,142)]
  catlabel = [19,19,19,0,1,19,19,2,3,4,19,19,19,5,6,7,8,9,10,11,12,13,14,15,19,19,16,17,18,19]
  Y = []
  X = []
  for picture_name in os.listdir(label_path) :
    image = Image.open(label_path+"/"+picture_name)
    y = np.array([[one_hot(catlabel[catcolor.index(image.getpixel((j,i)))]) for j in range(1024)] for i in range(512)])
    Y.append(y)
  
  print("finished loading labels")

  for picture_name in os.listdir(picture_path) :
    image = np.array(Image.open(picture_path+"/"+picture_name))/255.0
    X.append(image)
  num_data = len(X)
  X_test = X[:num_data//5]
  Y_test = Y[:num_data//5]
  X_train = X[num_data//5:]
  Y_train = Y[num_data//5:]

  print("finished loading pictures")

  return X_train,Y_train,X_test,Y_test
