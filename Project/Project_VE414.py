import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats

data = pd.read_csv("data_proj_414.csv")
xy = np.vstack([data["X"].values, data["Y"].values])
z = stats.gaussian_kde(xy)(xy)
# plt.scatter(data["X"].values, data["Y"].values, c=z, cmap="Set1", label="Route Density", alpha=0.02)
data_close = data[~data["Close"].isin([0])]
# plt.scatter(data_close["X"].values, data_close["Y"].values, c=data_close["Close"].values, cmap="Paired", alpha=1, label="Close")
# plt.scatter(data_close["X"].values, data_close["Y"].values,c=data_close["Close"].values+0.2*data_close["Far"].values,cmap="Set1",alpha=1,label="Close+0.2Far")
data_far = data[~data["Far"].isin([0])]
# plt.scatter(data_far["X"].values, data_far["Y"].values, c=data_far["Far"].values, cmap="Paired", alpha=0.02, label="Far")
# plt.scatter(data_far["X"].values, data_far["Y"].values, c=data_far["Close"].values+0.2*data_far["Far"].values,cmap="Paired",alpha=0.8)

def trip_test(data,index1,index2): 
    data_temp = data.copy()
    ls = list(np.arange(index1,index2))
    data_t = data_temp[data_temp['Trip'].isin(ls)]
    plt.scatter(data_t["X"].values, data_t["Y"].values,c=data_t['Trip'].values,cmap="twilight_shifted", alpha=1,label='Trip')
    plt.axis([0, 107, 0, 107])
    plt.title(f"Trip {index1}:{index2}")
    plt.legend()
    plt.colorbar()
    

def trip_plot(data_in):
    fig = plt.figure(figsize=(20,25),dpi=500)
    
    for i in range(1,7):
        plt.subplot(3,2,i)
        index1 = (i-1)*8+1
        index2 = i*8
        if (index2 == 48):
            index2 = index2+1
        xy = np.vstack([data["X"].values, data["Y"].values])
        z = stats.gaussian_kde(xy)(xy)
        plt.scatter(data["X"].values, data["Y"].values, c=z, cmap="Set1", label="Route Density", alpha=0.02)
        trip_test(data_in,index1,index2)
        plt.title(f"Trip_{i}",fontsize=12)
        print(f"{i} Trip finished!")
        
    fig.savefig("Trip")
    
trip_plot(data_far)
