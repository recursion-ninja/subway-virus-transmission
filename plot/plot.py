import csv
import pprint
import matplotlib.pyplot as plt 
import numpy as np 

# Setting up plotting data array and plotter
plotting_data = []
fig, ax = plt.subplots()


with open('./data/Results-Infection.csv') as f:
    csv_reader = csv.reader(f, delimiter=',')
    header = next(csv_reader)
    data = list(csv_reader)

    direction = data[0][1]

    inbound_contagious_data = []
    output = []
    output_upper = []
    output_lower = []

    for row in data:
        # New set of direction, push current direction data into plotting data
        # print(f'{direction} - {row[1]}')
        if direction != row[1]:
            plotting_data.append({
                'direction': direction,
                'data': {
                    'contagious_data': inbound_contagious_data,
                    'output': output,
                    'output_upper': output_upper,
                    'output_lower': output_lower
                }
            })
            direction = row[1]
            inbound_contagious_data = []
            output = []
            output_upper = []
            output_lower = []

        inbound_contagious_data.append(float(row[4]))
        output.append(float(row[5]))
        output_upper.append(float(row[6]))
        output_lower.append(float(row[7]))


    plotting_data.append({
        'direction': direction,
        'data': {
            'contagious_data': inbound_contagious_data,
            'output': output,
            'output_upper': output_upper,
            'output_lower': output_lower
        }
    })


for d in plotting_data:
    print(d['direction'])
    ax.set_title(d['direction'])
    x = d['data']['contagious_data']
    y = d['data']['output']
    y_lower = d['data']['output_lower']
    y_upper = d['data']['output_upper']
    ax.set_ylabel('Risk Metrics')
    ax.set_xlabel('Incoming Contagious')

    plt.plot(x,y, marker='x')
    plt.scatter(x,y_upper, marker='_')
    plt.scatter(x,y_lower, marker='_')
    plt.xticks(np.arange(min(x), max(x), 0.01))
    plt.yticks(np.arange(min(y_lower), max(y), 0.0015))

    plt.show()



# TODO: Plot for Station Limit and More Cars
# TODO: Parameterize script to allow for plotting of different columns?

# with open('./data/Results-Station-Limit.csv') as f:
# with open('./data/Results-More-Cars.csv') as f: