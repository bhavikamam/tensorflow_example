#example code to import a dataset using tensorflow using Python3.8
#using modified Iris dataset with float values

import pandas as pd, tensorflow as tf, numpy as np
tf.compat.v1.disable_v2_behavior() #tensorflow v2 is incompatble with several v1 instances.

#path="/home/xyz/"#enter your directory path.
#sys.argv[1] #disabled input option

filename=("iris.csv") #input data file in csv format comma separated
df=pd.read_csv(filename,usecols=[0,1,2,3],header=0) #using numeric columns of modified iris dataset
d=df.values
data=np.float32(d)
x=tf.compat.v1.placeholder(tf.compat.v1.float32,shape=(150,4)) #importing with tensorflow
x=data #data



