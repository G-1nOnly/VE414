from typing import ClassVar
import matplotlib.pyplot as plt
from numpy.lib import emath
import pandas as pd
import numpy as np
from scipy.stats.stats import weightedtau


def size_f(val_c, val_f):
    size = 5*val_c+val_f
    return size


def Random_append():
    x = np.random.uniform(0,1)
    if (x>=0.8):
        return True
    else:
        return False


def circle_scatter_add(data, x, y, val_c, val_f):
    data_new = data
    size = size_f(val_c, val_f)
    i_set = np.arange(0, size, 1)
    for _ in i_set:
        t = np.random.uniform(0, 1)*2*np.pi - np.pi
        len = np.random.uniform(0, 1)
        x_c = x+np.cos(t)*len
        y_c = y+np.sin(t)*len
        x_f = x+3*np.cos(t)*len
        y_f = y+3*np.sin(t)*len
        df_1 = pd.DataFrame(data=[[x_c, y_c, val_c, 0]], columns=["X", "Y", "Close", "Far"])
        data_new = data_new.append(df_1, ignore_index=True)
        df_2 = pd.DataFrame(data=[[x_f, y_f, 0, val_f]], columns=["X", "Y", "Close", "Far"])
        if(Random_append()):
            data_new = data_new.append(df_2, ignore_index=True)
    return data_new


def data_circle(data, size):
    data_new = data.copy()
    for i in range(size):
        data_new = circle_scatter_add(data_new, data_new.iloc[i]["X"], data_new.iloc[i]["Y"], data_new.iloc[i]["Close"], data_new.iloc[i]["Far"])
    return data_new


def density_plot(data):
    xx, yy = np.meshgrid(data["X"].values, data["Y"].values)
    temp = np.c_[xx.ravel(), yy.ravel()]
    z = []
    for x in temp:
        check = (data["X"].isin([x[0]])) & (data["Y"].isin([x[1]]))
        if (data[check].empty):
            num = 0
        else:
            num = data[check]["Weighted Number"]
        z.append(num)
    z = np.array(z, dtype=int)
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap="Paired")
    plt.colorbar()
    plt.axis([0, 107, 0, 107])
    plt.show()

def circle_scatter_add(data, x, y, val_c, val_f):
    data_new = data.copy()
    size = size_f(val_c, val_f)
    i_set = np.arange(0, size, 1)
    for _ in i_set:
        t = np.random.uniform(0, 1)*2*np.pi - np.pi
        len = np.abs(np.random.randn())
        if (len >= 3):
            len = 1
        else:
            len = len/3  
        x_c = x+np.cos(t)*len
        y_c = y+np.sin(t)*len
        x_f = x+3*np.cos(t)*len
        y_f = y+3*np.sin(t)*len
        df_1 = pd.DataFrame(data=[[x_c, y_c, val_c, 0]], columns=["X", "Y", "Close", "Far"])
        data_new = data_new.append(df_1, ignore_index=True)
        df_2 = pd.DataFrame(data=[[x_f, y_f, 0, val_f]], columns=["X", "Y", "Close", "Far"])
        if(Random_append()):
            data_new = data_new.append(df_2, ignore_index=True)
    return data_new

def circle_2D(x,y,r,size):
    ls_x = []
    ls_y = []
    i = 0
    while(i<size):
        x_t = x+np.random.uniform(-r,r)
        y_t = y+np.random.uniform(-r,r)
        if ((x_t-x)**2+(y_t-y)**2<=r**2):
            ls_x.append(x_t)
            ls_y.append(y_t)
            i = i+1
    
    plt.scatter(ls_x,ls_y)
    plt.show()
 
 
def circle_2D_data(data,x,y,val_c,val_f):
    
    data_new = data.copy()
    size_c = 10*val_c
    size_f = 2*val_f
    
    i = 0
    while(i<size_c):
        x_c = x+np.random.uniform(-1,1)
        y_c = y+np.random.uniform(-1,1)
        if ((x_c-x)**2+(y_c-y)**2<=1):
            df_temp = pd.DataFrame(data=[[x_c, y_c, val_c, 0]], columns=["X", "Y", "Close", "Far"])
            data_new = data_new.append(df_temp, ignore_index=True)
            i = i+1
    
    j = 0        
    while(j<size_f): 
        x_f = x+np.random.uniform(-3,3)
        y_f = y+np.random.uniform(-3,3)
        if ((x_f-x)**2+(y_f-y)**2 <= 9):
            df_temp = pd.DataFrame(data=[[x_f, y_f,0,val_f]], columns=["X", "Y", "Close", "Far"])
            data_new = data_new.append(df_temp, ignore_index=True)
            j = j+1 
    
    return data_new

def data_circle_2D(data, size):
    data_new = data.copy()
    for i in range(size):
        data_new = circle_2D_data(data_new, data_new.iloc[i]["X"], data_new.iloc[i]["Y"], data_new.iloc[i]["Close"], data_new.iloc[i]["Far"])
    return data_new


# data = pd.read_csv("data_proj_414.csv")
# data_close = data[~data["Close"].isin([0])][['X', 'Y', 'Close', 'Far']]
# data_now = data_circle_2D(data_close,len(data_close))
# data_now["Weighted Number"] = data_now["Close"]+0.2*data_now["Far"]
# plt.scatter(data_now["X"].values, data_now["Y"].values, c=data_now["Weighted Number"].values, cmap="Paired", alpha=1, label="Close")
# plt.show()
# density_plot(data_now)
 
def circle_normal_data(data, x, y, val_c, val_f):
    data_new = data
    size = size_f(val_c, val_f)
    i_set = np.arange(0, size, 1)
    for _ in i_set:
        t = np.random.uniform(0, 1)*2*np.pi - np.pi
        len = np.abs(np.random.randn())
        if (len >= 3):
            len = 1
        else:
            len = len/3         
        x_c = x+np.cos(t)*len
        y_c = y+np.sin(t)*len
        x_f = x+3*np.cos(t)*len
        y_f = y+3*np.sin(t)*len
        df_1 = pd.DataFrame(data=[[x_c, y_c, val_c, 0]], columns=["X", "Y", "Close", "Far"])
        data_new = data_new.append(df_1, ignore_index=True)
        df_2 = pd.DataFrame(data=[[x_f, y_f, 0, val_f]], columns=["X", "Y", "Close", "Far"])
        if(Random_append()):
            data_new = data_new.append(df_2, ignore_index=True)
    return data_new
   
def data_circle_normal(data, size):
    data_new = data.copy()
    for i in range(size):
        data_new = circle_normal_data(data_new, data_new.iloc[i]["X"], data_new.iloc[i]["Y"], data_new.iloc[i]["Close"], data_new.iloc[i]["Far"])
    return data_new

data = pd.read_csv("data_proj_414.csv")
data_close = data[~data["Close"].isin([0])][['X', 'Y', 'Close', 'Far']]
data_now = data_circle_normal(data_close,len(data_close))
data_now["Weighted Number"] = data_now["Close"]+0.2*data_now["Far"]
# plt.scatter(data_now["X"].values, data_now["Y"].values, c=data_now["Weighted Number"].values, cmap="Paired", alpha=1, label="Close")
# plt.show()
# density_plot(data_now)