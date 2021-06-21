import csv

# File to plot CSV data in the ./data folder.

# with open('./data/Results-Station-Limit.csv')
with open('./data/Results-Infection.csv') as f:
	csv_reader = csv.reader(f, delimiter=',')
	for row in csv_reader:
		print(row)