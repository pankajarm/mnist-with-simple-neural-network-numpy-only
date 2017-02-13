#  Copyright PankajMathur.com @2016
# This is a demnostration of buidling a simple neural network with three layers
# usign only numpy only and then training and testing mnist benchmark data with 97% accuracy
# huge thanks to for providing mnist data in csv format

from __future__ import division
import numpy as np

# neural network class definition
class SimpleNeuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the logistic sigmoid function
        # self.activation_function = lambda x: special.expit(x)
        self.activation_function = lambda x: 1 / ( 1+ np.exp(-x))
        pass

    
    # train the simple neural network
    def train(self, inputs_list, targets_list):

        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass

    
    # run the simple neural network
    def runner(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs



def pankax():

    # number of input, hidden and output nodes
	input_nodes = 784
	hidden_nodes = 500
	output_nodes = 10

	# learning rate
	learning_rate = 0.01

	# create instance of neural network
	n = SimpleNeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

	# Now, let's get the training data and Train the simple neural network with mnist data

	# load the mnist training data CSV file into a list
	training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
	training_data_list = training_data_file.readlines()
	print ("mnist train data loaded successfully")
	training_data_file.close()

	# epochs is the number of times the same training data set is used to train the neural network
	epochs = 5

	for e in range(epochs):

	    # we will go through all records in the training data set
	    record_index = 0
	    for record in training_data_list:

	        # then split the record by the ',' commas
	        all_values = record.split(',')

	        # then scale and shift the inputs
	        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

	        # let's create the target output values (all 0.01, except the desired label which is 0.99)
	        targets = np.zeros(output_nodes) + 0.01

	        # Do note, all_values[0] is the target label for this record
	        targets[int(all_values[0])] = 0.99

	        # Now, lets train our simple neural network
	        print ("training on {0} row...").format(record_index)
	        record_index += 1
	        n.train(inputs, targets)
	        
	        pass

	    print ("Training Done on {0} Epoch").format(e)
	    pass

	
	# Now, lets get the test data and Test the simple neueral network with mnist data

	# load the mnist test data CSV file into a list
	test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
	test_data_list = test_data_file.readlines()
	test_data_file.close()


	# let's keep a scorecard, to calculate how well the network performs
	scorecard = []

	# we will go through all the records in the test data set
	for record in test_data_list:

	    # then split the record by the ',' commas
	    all_values = record.split(',')

	    # do note correct answer is first value
	    correct_label = int(all_values[0])

	    # then scale and shift the inputs
	    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

	    # then run the simple neural network on test data 
	    outputs = n.runner(inputs)

	    # do note, the index of the highest value corresponds to the label
	    label = np.argmax(outputs)

	    # now, append correct or incorrect to scorecard list
	    if (label == correct_label):

	        # simple neural network's answer matches the correct answer, add 1 to scorecard
	        scorecard.append(1)

	    else:

	        # simple neural network's answer doesn't match correct answer, add 0 to scorecard
	        scorecard.append(0)
	        pass
	    
	    pass

	 
	# now, let's calculate the performance score which is the ratio of correct answers
	scorecard_array = np.asarray(scorecard)
	print ("performance = ", scorecard_array.sum() / scorecard_array.size)

	pass

if __name__ == '__main__':
    pankax()

