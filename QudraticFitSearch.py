#https://www.voxco.com/blog/quadratic-regression-calculator-definition-formula-and-calculation/
# importing packages and modules 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import r2_score 
#import scipy.stats as stats
#import csv

dataset = pd.read_csv('ball.csv') 
sns.scatterplot(data=dataset, x='time', y='height', hue='time') 
plt.title('time vs height of the ball') 
plt.xlabel('time') 
plt.ylabel('height') 
plt.show() 

# degree 2 polynomial fit or quadratic fit 
model = np.poly1d(np.polyfit(dataset['time'],dataset['height'], 2)) 

# polynomial line visualization 
polyline = np.linspace(0, 10, 100) 
plt.scatter(dataset['time'], dataset['height']) 
plt.plot(polyline, model(polyline)) 
plt.show() 

print(model) 

# r square metric 
print(r2_score(dataset['height'], 
			model(dataset['time']))) 
