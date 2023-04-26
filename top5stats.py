import csv

# IDEA: Get the top 5 performers for the following statistics: PPG, RPG, APG, FG%, FT%. The person that has the most
# top 5 appearances would statistically be the favorite to win the MVP award

# the column indices for the specified stats we want to check
columnsToCheck = [10, 17, 18, 19]

# dictionary to store the top 5 for each of the specified stats
top5Values = {j: [] for j in columnsToCheck} 

with open('NBA Stats 22-23 Top 50 PPG.csv', 'r') as file:
    # create a reader object
    reader = csv.reader(file)

    # iterate over each row in the file
    for i, row in enumerate(reader):
        # skip the header row
        if i == 0:
            continue

        # update the maximum values in each column of interest
        for j in columnsToCheck:
            value = float(row[j])
            if len(top5Values[j]) < 5:
                top5Values[j].append(value)
            elif value > min(top5Values[j]):
                top5Values[j].remove(min(top5Values[j]))
                top5Values[j].append(value)

# print the top 5 highest values in each column of interest
for j in columnsToCheck:  # replace with the actual column indices you're interested in
    print(f'Top 5 highest values in column {j}:')
    for value in sorted(top5Values[j], reverse=True):
        print(value)