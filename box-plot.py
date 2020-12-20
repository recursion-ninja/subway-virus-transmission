# Import libraries 
import matplotlib.pyplot as plt 
import numpy as np 
  
  
# Creating dataset 
np.random.seed(10) 
  
data_1 = np.genfromtxt('bbn.csv', delimiter=',')
data_2 = np.genfromtxt('bbt.csv', delimiter=',')
data_3 = np.genfromtxt('bbq.csv', delimiter=',')
#data_4 = np.genfromtxt('mbn.csv', delimiter=',')
#data_5 = np.genfromtxt('mbt.csv', delimiter=',')
#data_6 = np.genfromtxt('mbq.csv', delimiter=',')
data = [ data_1, data_2, data_3 ]
#data = [ data_4, data_5, data_6 ] 
  
labels = ['Control', 'More Trains', 'Fewer Passengers']

fig = plt.figure(figsize =(10, 7))
ax = fig.add_axes([0.15, 0.1, 0.7, 0.3])

# Creating axes instance 
#ax = fig.add_axes([0, 0, 1, 1]) 
  
# Creating plot 
bp = ax.boxplot(data, labels=labels, showfliers=False, notch=True) 
  
# show plot 
plt.show() 
