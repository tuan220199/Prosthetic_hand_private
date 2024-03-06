import csv
import matplotlib.pyplot as plt

# Open the CSV file in read mode
with open('recordingfiles/new_timer_9.csv', 'r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    
    # Initialize lists to store data from CSV
    y_values = []
    
    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Assuming one column of data in the CSV file
        # Append the element of each row to y_values
        y_values.append(float(row[0]))

# Generate x-values based on the index of each data point
x_values = list(range(1, len(y_values) + 1))

# Plot the graph as points
plt.scatter(x_values, y_values)
plt.xlabel('Index')
plt.ylabel('Y-axis label')
plt.title('Point Graph')
plt.show()
