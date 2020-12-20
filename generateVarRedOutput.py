import numpy as np
import pandas as pd
import sys
import subway_virus_simulation as svs
import csv

# arr1 = [1,2,3,4,5]
# arr2 = [2,4,6,8,10]
# arr3 = [5,10,15,20,25]

# arr_tot = zip(arr1, arr2, arr3)
# print(list(arr_tot))

# np.random.seed(618033988)
# print(np.random.randint(1,100),np.random.randint(1,100),np.random.randint(1,100),np.random.randint(1,100) )

# np.random.seed(618033988)
# print(np.random.randint(1,100),np.random.randint(1,100),np.random.randint(1,100),np.random.randint(1,100) )

sys.argv = ['L-Subway-Line.csv', 'Brooklyn-Bound.txt', 'Brooklyn', 'no-plot', '1000','time']
# sys.argv = ['L-Subway-Line.csv', 'Manhattan-Bound.txt', 'Manhattan', 'no-plot', '1000']
# sys.argv = ['L-Subway-Line.csv', 'Brooklyn-Bound-Extra.txt', 'Brooklyn', 'no-plot', '1000']
# sys.argv = ['L-Subway-Line.csv', 'Manhattan-Bound-Extra.txt', 'Manhattan', 'no-plot', '1000']
# sys.argv = ['L-Subway-Line.csv', 'Brooklyn-Bound.txt', 'Brooklyn', 'queue=1000', 'no-plot', '1000']
# sys.argv = ['L-Subway-Line.csv', 'Manhattan-Bound.txt', 'Manhattan', 'queue=1000', 'no-plot', '1000']
output, controlVarResults = svs.main()

sys.argv.append('anti')
output_anti, controlVarResults_anti = svs.main()

header = ['sim_results','sim_results_anti', 'total_overpacked_time', 'total_ride_time']

totalOverpackedTimes = [i[0] for i in controlVarResults]
totalRideTimes = [i[1] for i in controlVarResults]
all_data = zip(output, output_anti, totalOverpackedTimes, totalRideTimes)

with open("output.csv", "w+", newline='') as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(header)
    wr.writerows(all_data)

