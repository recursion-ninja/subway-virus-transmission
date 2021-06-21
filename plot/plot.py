import csv
import matplotlib.pyplot as plt 
import numpy as np 

# File to plot CSV data in the ./data folder.

# './data/Results-Infection.csv'
# ['1000', 'Brooklyn bound', 'Normal', '∞', '0.01', '0.00447', '0.00421', '0.00473']
# ['1000', 'Brooklyn bound', 'Normal', '∞', '0.02', '0.00816', '0.00773', '0.00860']
# ['1000', 'Brooklyn bound', 'Normal', '∞', '0.03', '0.01131', '0.01075', '0.01188']
# ['1000', 'Brooklyn bound', 'Normal', '∞', '0.04', '0.01399', '0.01331', '0.01468']
# ['1000', 'Brooklyn bound', 'Normal', '∞', '0.05', '0.01625', '0.01548', '0.01701']
# ['1000', 'Manhattan bound', 'Normal', '∞', '0.01', '0.06177', '0.06104', '0.06249']
# ['1000', 'Manhattan bound', 'Normal', '∞', '0.02', '0.09458', '0.09382', '0.09534']
# ['1000', 'Manhattan bound', 'Normal', '∞', '0.03', '0.11418', '0.11339', '0.11496']
# ['1000', 'Manhattan bound', 'Normal', '∞', '0.04', '0.12594', '0.12511', '0.12677']
# ['1000', 'Manhattan bound', 'Normal', '∞', '0.05', '0.13368', '0.13282', '0.13455']

# with open('./data/Results-Station-Limit.csv')
# with open('./data/Results-Infection.csv') as f:
# 	csv_reader = csv.reader(f, delimiter=',')
# 	for row in csv_reader:
# 		print(row)


y = [0.00447, 0.00816, 0.01131, 0.01399, 0.01625]
y_lower = [0.00421, 0.00773, 0.01075, 0.01331, 0.01548]
y_upper = [0.00473, 0.00860, 0.01188, 0.01468, 0.01701]
x = [0.01, 0.02, 0.03, 0.04, 0.05]


fig, ax = plt.subplots()

ax.set_title('Brooklyn Bound')
ax.set_ylabel('Risk Metrics')
ax.set_xlabel('Incoming Contagious')
plt.plot(x,y, marker='x')
plt.scatter(x,y_upper, marker='_')
plt.scatter(x,y_lower, marker='_')
plt.xticks(np.arange(min(x), max(x)+0.01, 0.01))
plt.yticks(np.arange(min(y_lower), max(y), 0.0015))

plt.show()
