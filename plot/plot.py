import csv
import pprint
import matplotlib.pyplot as plt 
import numpy as np 

# Setting up plotting data array and plotter
plotting_data = []

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
    plt.title(d['direction'])
    x = d['data']['contagious_data']
    y = d['data']['output']
    y_lower = d['data']['output_lower']
    y_upper = d['data']['output_upper']
    plt.ylabel('Risk Metrics')
    plt.xlabel('Incoming Contagious')

    plt.plot(x,y, marker='x')
    plt.scatter(x,y_upper, marker='_')
    plt.scatter(x,y_lower, marker='_')
    plt.xticks(np.arange(min(x), max(x), 0.01)) # TODO: Determine X/Y Ticks for both directions
    plt.yticks(np.arange(min(y_lower), max(y), 0.0015))

    plt.show()


#
# Plot 2nd file
#

plotting_data = []

with open('./data/Results-Station-Limit.csv') as f:
    csv_reader = csv.reader(f, delimiter=',')
    header = next(csv_reader)
    data = list(csv_reader)

    direction = data[0][1]

    station_limit = []
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
                    'station_limit': station_limit,
                    'output': output,
                    'output_upper': output_upper,
                    'output_lower': output_lower
                }
            })
            direction = row[1]
            station_limit = []
            output = []
            output_upper = []
            output_lower = []

        station_limit.append(float(row[3]))
        output.append(float(row[5]))
        output_upper.append(float(row[6]))
        output_lower.append(float(row[7]))


    plotting_data.append({
        'direction': direction,
        'data': {
            'station_limit': station_limit,
            'output': output,
            'output_upper': output_upper,
            'output_lower': output_lower
        }
    })


for d in plotting_data:
    plt.title(d['direction'])
    x = d['data']['station_limit']
    y = d['data']['output']
    y_lower = d['data']['output_lower']
    y_upper = d['data']['output_upper']
    plt.ylabel('Risk Metrics')
    plt.xlabel('Station Limit')

    plt.plot(x,y, marker='x')
    plt.scatter(x,y_upper, marker='_')
    plt.scatter(x,y_lower, marker='_')
    plt.xticks(np.arange(min(x), max(x), 10))
    plt.yticks(np.arange(min(y_lower), max(y))) # TODO: Determine Y ticks

    plt.show()

#
# Plot 3
#

plotting_data = []

with open('./data/Results-More-Cars.csv') as f:
    csv_reader = csv.reader(f, delimiter=',')
    header = next(csv_reader)
    data = list(csv_reader)

    direction = data[0][1]

    cars = []
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
                    'cars': cars,
                    'output': output,
                    'output_upper': output_upper,
                    'output_lower': output_lower
                }
            })
            direction = row[1]
            cars = []
            output = []
            output_upper = []
            output_lower = []

        cars.append(float(row[3]))
        output.append(float(row[6]))
        output_upper.append(float(row[7]))
        output_lower.append(float(row[8]))


    plotting_data.append({
        'direction': direction,
        'data': {
            'cars': cars,
            'output': output,
            'output_upper': output_upper,
            'output_lower': output_lower
        }
    })


for d in plotting_data:
    plt.title(d['direction'])
    x = d['data']['cars']
    y = d['data']['output']
    y_lower = d['data']['output_lower']
    y_upper = d['data']['output_upper']
    plt.ylabel('Risk Metrics')
    plt.xlabel('Station Limit')

    plt.plot(x,y, marker='x')
    plt.scatter(x,y_upper, marker='_')
    plt.scatter(x,y_lower, marker='_')
    plt.xticks(np.arange(min(x), max(x), 1))
    plt.yticks(np.arange(min(y_lower), max(y))) # TODO: Determine Y ticks

    plt.show()

# TODO: Plot for Station Limit and More Cars
# TODO: Parameterize script to allow for plotting of different columns?