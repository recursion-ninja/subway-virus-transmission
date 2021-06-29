import csv
import pprint
import matplotlib.pyplot as plt 
import numpy as np 

# Setting up plotting data array and plotter
plotting_data = []
saveDir = 'pngs'
shouldShow = True
shouldSave = False

def showFigure(filePath, title, columnIndex, x_step, y_step, y_max):
    data_set = []
    with open(filePath) as f:
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
                data_set.append({
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

            inbound_contagious_data.append(float(row[columnIndex]))
            output.append(float(row[5]))
            output_upper.append(float(row[6]))
            output_lower.append(float(row[7]))


        data_set.append({
            'direction': direction,
            'data': {
                'contagious_data': inbound_contagious_data,
                'output': output,
                'output_upper': output_upper,
                'output_lower': output_lower
            }
        })

#    fig = plt.figure()
#    plt.clf()
#    plt.cla()
#    plt.close()
    fig, ax = plt.subplots()
#    fig.clear()
#    ax.clear()
    for d in data_set:
        x = d['data']['contagious_data']
        y = d['data']['output']
        y_lower = d['data']['output_lower']
        y_upper = d['data']['output_upper']
        ax.plot(x,y, marker='x', label = d['direction'])
        ax.scatter(x,y_upper, marker='_')
        ax.scatter(x,y_lower, marker='_')

    ax.set_ylabel('Risk Metrics')
    ax.set_xlabel('Incoming Contagious')
    ax.set_xticks(np.arange(min(x), max(x) + x_step, x_step))
    ax.set_yticks(np.arange(0, y_max, y_step))
    ax.set_title(title)
    fig.legend()

    if shouldShow:
        fig.show()
    if shouldSave:
        fig.savefig(saveDir + '/' + title + '.png', dpi=150)

#    ax.clear()
#    fig.clear()

#def main():
#    showFigure(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
#
#if __name__ == "__main__":
#    main()

        

    
#print('About to generate figure')

#showFigure( './data/Results-Infection.csv'
#          , 'Contagiousness of the Incoming Population'
#          , 4
#          , 0.01
#          , 0.01
#          , 0.15
#          )

#showFigure( './data/Results-Station-Limit.csv'
#          , 'Station Entry Limiting'
#          , 3
#          , 10
#          , 0.001
#          , 0.006
#          )


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
    x = d['data']['contagious_data']
    y = d['data']['output']
    y_lower = d['data']['output_lower']
    y_upper = d['data']['output_upper']
    plt.ylabel('Risk Metrics')
    plt.xlabel('Incoming Contagious')

    plt.plot(x,y, marker='x', label = d['direction'])
    plt.scatter(x,y_upper, marker='_')
    plt.scatter(x,y_lower, marker='_')

x_step = 0.01
plt.xticks(np.arange(min(x), max(x) + x_step, x_step))
y_step = 0.01
y_max  = 0.15
        
plt.yticks(np.arange(0, y_max, y_step))
plt.title('Contagiousness of the Incoming Population')
plt.legend()

if shouldShow:
    plt.show()
if shouldSave:
    plt.savefig(saveDir + '/' + d['direction'] + ' ' + 'Infection.png', dpi=150)



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

    plt.plot(x,y, marker='x', label = d['direction'])
    plt.scatter(x,y_upper, marker='_')
    plt.scatter(x,y_lower, marker='_')
        
x_step = 10
y_step = 0.001
y_max  = 0.006

plt.ylabel('Risk Metrics')
plt.xlabel('Platform Limit')
plt.xticks(np.arange(min(x), max(x) + x_step, x_step))
plt.yticks(np.arange(0, y_max, y_step))
plt.title('Controling Platform Capacity Via Limiting Entry')
plt.legend()

if shouldShow:
    plt.show()
if shouldSave:
    plt.savefig(saveDir + '/' + d['direction'] + ' ' + 'Station.png', dpi=150)



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

    plt.plot(x,y, marker='x', label = d['direction'])
    plt.scatter(x,y_upper, marker='_')
    plt.scatter(x,y_lower, marker='_')
        
x_step = 2
y_step = 0.01
y_max  = 0.07

plt.ylabel('Risk Metrics')
plt.xlabel('Train Car Quantity')
plt.xticks(np.arange(min(x), max(x) + x_step, x_step))
plt.yticks(np.arange(0, y_max + y_step, y_step))
plt.title('Providing Space Via Additional Traincars')
plt.legend()

if shouldShow:
    plt.show()
if shouldSave:
    plt.savefig(saveDir + '/' + d['direction'] + ' ' + 'Cars.png', dpi=150)

