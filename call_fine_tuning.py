import theano
import theano.tensor as T
import numpy as np
import numpy
from fine_tuning import fine_tuning
rng = numpy.random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# simulation data
training_steps = 1000
simulate_data_num = 50
num_batch = 3
lr = 0.01

ini_value_w1 = rng.randn(10,8)
ini_value_w2 = rng.randn(8,6)
ini_value_w3 = rng.randn(6,4)
ini_value_w4 = rng.randn(4,2)
w_list_3_layer = [ini_value_w1, ini_value_w2, ini_value_w3]
w_list_4_layer = [ini_value_w1, ini_value_w2, ini_value_w3, ini_value_w4]

ini_value_b1 = rng.randn(8,1)
ini_value_b2 = rng.randn(6,1)
ini_value_b3 = rng.randn(4,1)
ini_value_b4 = rng.randn(2,1)
ini_value_b5 = rng.randn(4,1)
ini_value_b6 = rng.randn(6,1)
ini_value_b7 = rng.randn(8,1)
ini_value_b8 = rng.randn(10,1)
b_list_3_layer = [ini_value_b1, ini_value_b2, ini_value_b3, ini_value_b6, ini_value_b7, ini_value_b8]
b_list_4_layer = [ini_value_b1, ini_value_b2, ini_value_b3, ini_value_b4, ini_value_b5, ini_value_b6, ini_value_b7, ini_value_b8]

simulate_data = rng.randn(10,simulate_data_num)  # 50 data points

total_cost_record_3_lay, w_list_output_3_lay, b_list_output_3_lay = fine_tuning(w_list_3_layer, b_list_3_layer, simulate_data, num_batch, training_steps, lr)
total_cost_record_4_lay, w_list_output_4_lay, b_list_output_4_lay = fine_tuning(w_list_4_layer, b_list_4_layer, simulate_data, num_batch, training_steps, lr)



fig, ax = plt.subplots()
ax.scatter(range(len(total_cost_record_3_lay)),total_cost_record_3_lay)
ax.scatter(range(len(total_cost_record_4_lay)),total_cost_record_4_lay)
plt.show()