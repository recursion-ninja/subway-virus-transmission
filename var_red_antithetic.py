import numpy as np
import pandas as pd
import subprocess
import sys
from subway_virus_simulation import confidenceIntervalString
import csv


#### HOW TO USE THIS FILE #####
# Call this file on the command-line with python3 command and #-of-replications arguement.
# Example: 'python3 var_red_antithetic.py 2'
##########  END HOW-TO  #######

def main():

    replications = sys.argv[1]
    process = subprocess.run(['python3', 'subway_virus_simulation.py', 'L-Subway-Line.csv', 'Brooklyn-Bound.txt', str(replications), 'record'])
    process = subprocess.run(['python3', 'subway_virus_simulation.py', 'L-Subway-Line.csv', 'Brooklyn-Bound.txt', str(replications), 'record', 'anti'])

    output = pd.read_csv("output.csv", header=None)
    mean_output = output.describe().loc['mean']

    #print(output)
    print("Batch 1 Outcome: ", confidenceIntervalString(output.iloc[0,:]))
    print("Batch 2 Outcome: ", confidenceIntervalString(output.iloc[1,:]))
    print("Antithetic Combined Batch Outcome: ", confidenceIntervalString(mean_output))

    # with open('output.csv', 'w+') as f:
    #     print("")

if __name__ == "__main__":
    main()
