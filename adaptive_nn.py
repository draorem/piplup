import math
import numpy as np

#NOTICE BOARD
#add to class prop (else block - create neurons in second hidden layer and beyond)
#add input for user data
#add extra parameters of iteration count and epoch count (saving current training progress for optional export)
#eventually force valid inputs for initial setup, integers + reset if invalid

layer_count = input("How many layers? There must be at least 2, for input and output. > ")
mu = 0 #mean of randomized values (weights and biases)
sigma = 0.1 #standard deviation of randomized values (weights and biases)
neuron_outline = [] #neuron count by layer
w = [] #weight-ing list
b = [] #bias list
neuron_values = [] #actual numerical values of neurons, added to through forward propagation

x = [4,5] #input(s) for a single data sample

for i in range(int(layer_count)): #user input for neuron count in each layer
    if i == 0:
        neuron_outline.append(int(input("Neurons in layer {} (input): ".format(i))))
    elif i == (int(layer_count) - 1):
        neuron_outline.append(int(input("Neurons in layer {} (output): ".format(i))))
        w.append(np.random.default_rng().normal(mu,sigma,(int(neuron_outline[i])*int(neuron_outline[i-1]))))
        b.append(np.random.default_rng().normal(mu,sigma,1))
    else:
        neuron_outline.append(int(input("Neurons in layer {}: ".format(i))))
        w.append(np.random.default_rng().normal(mu,sigma,(int(neuron_outline[i])*int(neuron_outline[i-1]))))
        b.append(np.random.default_rng().normal(mu,sigma,1))

print("Neuron outline: ",neuron_outline)

for i in range(len(w)):
    print("Weights: ",w[i])
    print("Biases: ",b[i])
    
#add to a_labels with addition of more activation functions
#0 - ReLU, 1 - sigmoid    
a_labels = [] #identification of activation functions for hidden layer(s) + output(s)
if int(layer_count) > 2:
    print("What activation function should be applied for the hidden layers?")
    a_labels_mark = input("0 for ReLU, 1 for softmax > ")
    if int(a_labels_mark) == 0:
        for i in range(len(neuron_outline)-2):
            a_labels.append(0)
    else:
        for i in range(len(neuron_outline)-2):
            a_labels.append(1)
print("What activation function should be applied for the output layer?")
a_labels_mark = input("0 for ReLU, 1 for softmax > ")
if int(a_labels_mark) == 0:
    a_labels.append(0)
else:
    a_labels.append(1)
    
print(a_labels) #identification of activation functions for hidden layer(s) + output(s)

c_neuron_outline = [] #cumulative frequency of neurons, final value == total # of neurons
c_neuron_outline = [sum(neuron_outline[0:x]) for x in range(1, len(neuron_outline)+1)]
    
for i in range(len(x)): #inputs
    neuron_values.append(x[i]) #adding to list as first neurons

print("C_neuron_outline: ",c_neuron_outline)
print("Neuron values: ",neuron_values)

for i in range(len(c_neuron_outline)): #to find previous layer index (for neuron count)
    index_search = [] #index, how many neurons at said layer
    if c_neuron_outline[len(c_neuron_outline)-i-1] <= len(neuron_values):
        index_count = neuron_outline[len(c_neuron_outline)-i-1] #number of neurons in index
        index = neuron_outline.index(c_neuron_outline[len(c_neuron_outline)-i-1]) #index to be used in neuron_outline
        index_search.append(index)
        index_search.append(index_count)
        
print("index, neuron count: ",index_search)
print("neuron count: ",c_neuron_outline[-1])

class neuron:
    def init(input, weight, bias):
        return (input * weight) + bias
    def forward(input, ReLU, softmax): #1 if activation function is active, 0 if not
        return (ReLU * max(0,input)) + (softmax * ((1 + math.exp(-1 * input)) ** -1))
    def backward(input, ReLU, softmax): #1 if activation function is active, 0 if not
        return (ReLU * (1 - ((1 / input)*(min(0,input))))) + (softmax * ((math.exp(-1 * input) / ((1 + math.exp(-1 * input)) ** 2))))

class prop: #have not tested this yet
    def forward(input, weight, bias):
        for i in range(c_neuron_outline[-1] - len(neuron_values)): #remaining neurons to code
            if i < neuron_outline[1]: #neurons in first hidden layer
                pre_neuron = 0
                for j in range(neuron_outline[i] * neuron_outline[i+1]):
                    pre_neuron += neuron.init(neuron_values[j % neuron_outline[0]],w[j],0) #weights only
                layer_neuron = neuron.forward(neuron.init(pre_neuron,1,b[0]),1-a_labels[0],a_labels[0]) #adding bias + activation
                neuron_values.append(layer_neuron)
            else: #neurons in second layer and beyond
                pre_neuron = 0
                #wip
            i += 1
        print("Forward pass:") #also haven't tested this yet
        fp_neuron_count = 0 
        for i in range(len(neuron_outline)): #one line for each layer
            for j in range(neuron_outline[i]): #repeats for each neuron
                print("Layer {}: ".format(i),neuron_values[fp_neuron_count])
                fp_neuron_count += 1
    def backward(myprecioustime, mypreciouseffort): #wip
        trashcanofdespair = []
        trashcanofdespair.append(myprecioustime,mypreciouseffort)

#class error:
    #def forward(target, output, output_neuron_count):
        #re = 0
        #for i in range(output_neuron_count):
            #re += (0.5*((target[i]-output[i]) ** 2))
        #return re
    #def backward(): #???