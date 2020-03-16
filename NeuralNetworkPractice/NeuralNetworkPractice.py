
#A basic perceptron.


import numpy

import numpy as np

#So we return the value of the sigmoid function, which will give us a number between 0 and 1 as a means of normalization.
def sigmoid(x):
	return 1/ (1+ np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x)

training_inputs = np.array([[0,0,1],
						   [1,1,1],
						   [1,0,1],
						   [0,0,1]])

#This is a row but notice the t, so we can transpose it into a 4*1 matrix. 
training_outputs = np.array([[0,1,1,0]]).T

#Now we initialize our weights, we'll use random values for that. 
#Seeding ensures we get same random as tut
np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) -1 

print("Weights : {}".format(synaptic_weights))

for iteration in range(100000):
	input_layer = training_inputs
	outputs = sigmoid(np.dot(input_layer, synaptic_weights))
	error = training_outputs - outputs
	#We see the sigmoid gradient coming into use there. 
	adjustment = error * sigmoid_derivative(outputs)
	synaptic_weights += np.dot(input_layer.T, adjustment)



#Should be more accurate, coming out as 2 etcccc 
print("output after training : {} ".format(outputs))

#So now we've done a it we follow a process(Backprop)
# 1 Take the inputs from training example and put them through our formula to get neurons output.
# 2 Calculate the error, which is diff between the output we got and actual output.
# 3 Now using this(sort of similar to like k-medoids etc) we adjust the weights. 
# 4 Repeat a number of times, tutorial suggests 20,000.
#Refer to notes for the math.