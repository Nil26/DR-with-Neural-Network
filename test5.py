import theano
import theano.tensor as T
import numpy as np
import numpy
rng = numpy.random

# Assume the input data is a matrix of size(data_num,data_dim)
# Assume the input data has already been normalized (zero mean and unit variance)
# this is written for one batch, one layer


'''
b1 = T.matrix("b1")
b2 = T.matrix("b2")
b3 = T.matrix("b3")
b4 = T.matrix("b4")
b5 = T.matrix("b5")
b6 = T.matrix("b6")

w1 = T.matrix('w1')
w2 = T.matrix('w2')
w3 = T.matrix('w3')
w4 = w1.T
w5 = w2.T
w6 = w3.T
'''
data = T.matrix("data")

training_steps = 1000
simulate_data_num = 50
lr = 0.01

ini_value_w1 = rng.randn(10,8)
ini_value_w2 = rng.randn(8,6)
ini_value_w3 = rng.randn(6,4)

ini_value_b1 = np.tile(rng.randn(8,1),(1,simulate_data_num))
ini_value_b2 = np.tile(rng.randn(6,1),(1,simulate_data_num))
ini_value_b3 = np.tile(rng.randn(4,1),(1,simulate_data_num))
ini_value_b4 = np.tile(rng.randn(6,1),(1,simulate_data_num))
ini_value_b5 = np.tile(rng.randn(8,1),(1,simulate_data_num))
ini_value_b6 = np.tile(rng.randn(10,1),(1,simulate_data_num))

w1 = theano.shared(ini_value_w1, name="w1")
w2 = theano.shared(ini_value_w2, name="w2")
w3 = theano.shared(ini_value_w3, name="w3")
w4 = w3.T
w5 = w2.T
w6 = w1.T

b1 = theano.shared(ini_value_b1, name="b1")
b2 = theano.shared(ini_value_b2, name="b2")
b3 = theano.shared(ini_value_b3, name="b3")
b4 = theano.shared(ini_value_b4, name="b4")
b5 = theano.shared(ini_value_b5, name="b5")
b6 = theano.shared(ini_value_b6, name="b6")

para = [b1, b2, b3, b4, b5, b6, w1, w2, w3]

step1_result = T.dot(w1.T,data) + b1 > 0
step2_result = T.dot(w2.T,step1_result) + b2 > 0
step3_result = T.dot(w3.T,step2_result) + b3 > 0
step4_result = T.dot(w4.T,step3_result) + b4 > 0
step5_result = T.dot(w5.T,step4_result) + b5 > 0
result = T.dot(w6.T,step5_result) + b6

diff = result - data # Cross-entropy loss function
cost = (diff ** 2).mean() # no regulization
gw1, gw2, gw3, gb1, gb2, gb3, gb4, gb5, gb6 = T.grad(cost, para)

# Compile
train = theano.function(
          inputs=[data],
          outputs=[cost],
          updates=((w1, w1 - lr*gw1),(w2, w2 - lr*gw2), \
          	(w3, w3 - lr*gw3), (b1, b1 - lr*gb1), \
          	(b2, b2 - lr*gb2), (b3, b3 - lr*gb3), \
          	(b4, b4 - lr*gb4), (b5, b5 - lr*gb5), \
          	(b6, b6 - lr*gb6)))

predict = theano.function(inputs=[data], outputs=result)

# Train

simulate_data = rng.randn(10,simulate_data_num)  # 50 data points


input = simulate_data

cost_record = []
for i in range(training_steps):
	print("iteration:",i)
	cost = train(input)
	cost_record.append(cost)

print(cost_record)
print(cost_record[-1])

'''
print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))
'''