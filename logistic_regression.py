import csv
import math
num_steps = 1000
step_size = .00001

y_vals = []
thetas = []
num_excluded = 1

def sigmoid_func(z):
    denominator = 1 + math.exp(-z)
    return (1/denominator)

def initialize_list(length):
    new_list = []
    for i in range(length):
        new_list.append(0)
    return new_list

def calculate_z(thetas, xis):
    z = 0
    for theta, xi in zip(thetas, xis):
        z += (theta * int(xi))
    return z

# Training
with open("heart-train.csv") as f:
    csvreader = csv.reader(f)
    header = next(csvreader)
    # Find if there's a demographic section and exclude it (alongside y-values) in xi data accordingly
    if 'Demographic' in header:
        num_excluded = 2

    # Initialize thetas as 0, including an extra x0
    thetas = initialize_list(len(header) - num_excluded + 1)
    
    # Create list of training examples
    training_examples = []
    for row in csvreader:
        training_examples.append(row)

    # Repeat process many times
    for i in range(num_steps):
        # Initialize gradients
        gradients = initialize_list(len(header) - num_excluded + 1)
        
        # Go through every example
        for row in training_examples:
            y_val = int(row[len(row) - 1])

            # Add 1 at the beginning
            new_row = [1]
            new_row += row[:-num_excluded]

            z = calculate_z(thetas, new_row)

            # Go through every xi, including new added "x0"
            for j in range(len(new_row)):
                xi_val = int(new_row[j])
                # Update gradients[j]
                gradients[j] += (xi_val * (y_val - sigmoid_func(z)))
            
        # Update thetas
        for y in range(len(thetas)):
            gradient = gradients[y]
            thetas[y] += (step_size * gradient)
                
# Now we can test
guesses = []
real_y_vals = []
with open("heart-test.csv") as f:
    csvreader = csv.reader(f)
    header = next(csvreader)
    for row in csvreader:
        # Add 1 to the start of every row
        new_row = [1]
        new_row += row[:-num_excluded]
        z = calculate_z(thetas, new_row)
        guess = int(sigmoid_func(z) >= .5)
        guesses.append(guess)
        real_y_vals.append(int(row[-1]))

    num_correct = 0
    # Find accuracy
    for guess, y_val in zip(guesses, real_y_vals):
        if guess == y_val:
            num_correct += 1
    print(num_correct/len(guesses))