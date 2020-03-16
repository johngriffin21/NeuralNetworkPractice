import numpy as np

class NeuralNetwork():

	def __init__(self):
		np.random.seed(1)
		self.synaptic_weights = 2 * np.random.random((3,1)) - 1

	def sigmoid(self, x):
		return 1/ (1 + np.exp(-x))

	def sigmoid_derivative(self, x):
		return x * (1-x)

	def train(self, training_inputs, training_outputs, training_iterations):
		for iteration in range(training_iterations):
			output = self.think(training_inputs)
			error  = training_outputs - output
			adjustments = np.dot(training_inputs.T, error * (self.sigmoid_derivative(output)))
			self.synaptic_weight += adjustments

	def think(self,inputs):
		inputs = inputs.astype(float)
		output = self.sigmoid(np.dot(input, self.synaptic_weights))
		return output

if __name__ == "__main__":
	neural_network = NeuralNetwork()

	print("Random synaptic weights {}".format(neural_network.synaptic_weights))

	training_inputs = np.array([[0,0,1],
						   [1,1,1],
						   [1,0,1],
						   [0,0,1]])

	training_outputs = np.array([[0,1,1,0]]).T

	neural_network.train(training_inputs, training_outputs, 100000)

	print("Weights after training : {}".format(neural_network.synaptic_weights))

	A = str(input("Int 1:  "))
	B = str(input("Int 2:  "))
	C = str(input("Int 3:  "))

	print("Your input data is : {} {} {} ".format(A, B, C))

	print("Your new output data is {}".foramt(neural_network.think(np.array([A,B,C]))))


		
