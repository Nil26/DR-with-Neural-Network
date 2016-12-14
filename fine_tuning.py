import theano
import theano.tensor as T
import numpy as np

# Assume the input data is a matrix of size(data_num,data_dim)
# Assume the input data has already been normalized (zero mean and unit variance)
# this is written for one batch, one layer
def fine_tuning(w_list, b_list, data, num_batch, training_steps, lr):
     (data_dim, data_num) = data.shape
     one_batch_size = int(data_num / num_batch)
     total_cost_record = []

     if len(w_list) == 6:
          for count in range(num_batch-1):
               batch_data = data[:,count*one_batch_size:(count+1)*one_batch_size]
               cost_record, w_list_output, b_list_output = fine_tuning_3_one_batch(w_list, b_list, batch_data, training_steps, lr)
               total_cost_record = total_cost_record + cost_record
               w_list = w_list_output
               b_list = b_list_output

          # last batch
          batch_data = data[:,(num_batch-1)*one_batch_size:]
          cost_record, w_list_output, b_list_output = fine_tuning_3_one_batch(w_list, b_list, batch_data, training_steps, lr)
          total_cost_record = total_cost_record + cost_record
     elif len(w_list) == 8:
          for count in range(num_batch-1):
               batch_data = data[:,count*one_batch_size:(count+1)*one_batch_size]
               cost_record, w_list_output, b_list_output = fine_tuning_4_one_batch(w_list, b_list, batch_data, training_steps, lr)
               total_cost_record = total_cost_record + cost_record
               w_list = w_list_output
               b_list = b_list_output

          # last batch
          batch_data = data[:,(num_batch-1)*one_batch_size:]
          cost_record, w_list_output, b_list_output = fine_tuning_4_one_batch(w_list, b_list, batch_data, training_steps, lr)
          total_cost_record = total_cost_record + cost_record
     else:
          raise('only support 3 or 4 layer network')


     return total_cost_record, w_list_output, b_list_output


def fine_tuning_3_one_batch(w_list, b_list, batch_data, training_steps, lr):

     (batch_data_dim, batch_data_num) = batch_data.shape
     ini_value_w1 = w_list[0]
     ini_value_w2 = w_list[1]
     ini_value_w3 = w_list[2]
     ini_value_w4 = w_list[3]
     ini_value_w5 = w_list[4]
     ini_value_w6 = w_list[5]
     ini_value_b1 = np.tile(b_list[0],(1,batch_data_num))
     ini_value_b2 = np.tile(b_list[1],(1,batch_data_num))
     ini_value_b3 = np.tile(b_list[2],(1,batch_data_num))
     ini_value_b4 = np.tile(b_list[3],(1,batch_data_num))
     ini_value_b5 = np.tile(b_list[4],(1,batch_data_num))
     ini_value_b6 = np.tile(b_list[5],(1,batch_data_num))

     data_sym = T.matrix("data_sym")

     w1 = theano.shared(ini_value_w1, name="w1")
     w2 = theano.shared(ini_value_w2, name="w2")
     w3 = theano.shared(ini_value_w3, name="w3")
     w4 = theano.shared(ini_value_w4, name="w4")
     w5 = theano.shared(ini_value_w5, name="w5")
     w6 = theano.shared(ini_value_w6, name="w6")

     b1 = theano.shared(ini_value_b1, name="b1")
     b2 = theano.shared(ini_value_b2, name="b2")
     b3 = theano.shared(ini_value_b3, name="b3")
     b4 = theano.shared(ini_value_b4, name="b4")
     b5 = theano.shared(ini_value_b5, name="b5")
     b6 = theano.shared(ini_value_b6, name="b6")
     para = [b1, b2, b3, b4, b5, b6, w1, w2, w3, w4, w5, w6]

     step1_result = T.dot(T.transpose(w1),data_sym) + b1 > 0
     step2_result = T.dot(T.transpose(w2),step1_result) + b2 > 0
     step3_result = T.dot(T.transpose(w3),step2_result) + b3 > 0
     step4_result = T.dot(T.transpose(w4),step3_result) + b4 > 0
     step5_result = T.dot(T.transpose(w5),step4_result) + b5 > 0
     result = T.dot(T.transpose(w6),step5_result) + b6



     diff = result - data_sym # Cross-entropy loss function
     cost = (diff ** 2).mean() # no regulization, this will return the mean of the matrix
     gb1, gb2, gb3, gb4, gb5, gb6, gw1, gw2, gw3, gw4, gw5, gw6 = T.grad(cost, para)

     # Compile

     train = theano.function(
               inputs=[data_sym],
               outputs=[cost],
               updates=((w1, w1 - lr*gw1),(w2, w2 - lr*gw2), \
               	(w3, w3 - lr*gw3), (w4, w4 - lr*gw4), (w5, w5 - lr*gw5), \
                    (w6, w6 - lr*gw6), (b1, b1 - lr*gb1), \
               	(b2, b2 - lr*gb2), (b3, b3 - lr*gb3), \
               	(b4, b4 - lr*gb4), (b5, b5 - lr*gb5), \
               	(b6, b6 - lr*gb6)))

     #predict = theano.function(inputs=[data], outputs=result)

     # Train
     cost_record = []
     for i in range(training_steps):
     	cost = train(batch_data)
     	cost_record.append(cost)
     w_list_output = [w1.get_value(),w2.get_value(),w3.get_value(),w4.get_value(),w5.get_value(),w6.get_value()]
     b_list_output = [b1.get_value()[:,[0]],b2.get_value()[:,[0]],b3.get_value()[:,[0]],b4.get_value()[:,[0]],b5.get_value()[:,[0]],b6.get_value()[:,[0]]]
     return cost_record, w_list_output, b_list_output


def fine_tuning_4_one_batch(w_list, b_list, batch_data, training_steps, lr):

     (batch_data_dim, batch_data_num) = batch_data.shape
     ini_value_w1 = w_list[0]
     ini_value_w2 = w_list[1]
     ini_value_w3 = w_list[2]
     ini_value_w4 = w_list[3]
     ini_value_w5 = w_list[4]
     ini_value_w6 = w_list[5]
     ini_value_w7 = w_list[6]
     ini_value_w8 = w_list[7]
     ini_value_b1 = np.tile(b_list[0],(1,batch_data_num))
     ini_value_b2 = np.tile(b_list[1],(1,batch_data_num))
     ini_value_b3 = np.tile(b_list[2],(1,batch_data_num))
     ini_value_b4 = np.tile(b_list[3],(1,batch_data_num))
     ini_value_b5 = np.tile(b_list[4],(1,batch_data_num))
     ini_value_b6 = np.tile(b_list[5],(1,batch_data_num))
     ini_value_b7 = np.tile(b_list[6],(1,batch_data_num))
     ini_value_b8 = np.tile(b_list[7],(1,batch_data_num))

     data_sym = T.matrix("data_sym")

     w1 = theano.shared(ini_value_w1, name="w1")
     w2 = theano.shared(ini_value_w2, name="w2")
     w3 = theano.shared(ini_value_w3, name="w3")
     w4 = theano.shared(ini_value_w4, name="w4")
     w5 = theano.shared(ini_value_w5, name="w5")
     w6 = theano.shared(ini_value_w6, name="w6")
     w7 = theano.shared(ini_value_w7, name="w7")
     w8 = theano.shared(ini_value_w8, name="w8")


     b1 = theano.shared(ini_value_b1, name="b1")
     b2 = theano.shared(ini_value_b2, name="b2")
     b3 = theano.shared(ini_value_b3, name="b3")
     b4 = theano.shared(ini_value_b4, name="b4")
     b5 = theano.shared(ini_value_b5, name="b5")
     b6 = theano.shared(ini_value_b6, name="b6")
     b7 = theano.shared(ini_value_b7, name="b7")
     b8 = theano.shared(ini_value_b8, name="b8")
     para = [b1, b2, b3, b4, b5, b6, b7, b8, w1, w2, w3, w4, w5, w6, w7, w8]

     step1_result = T.dot(T.transpose(w1),data_sym) + b1 > 0
     step2_result = T.dot(T.transpose(w2),step1_result) + b2 > 0
     step3_result = T.dot(T.transpose(w3),step2_result) + b3 > 0
     step4_result = T.dot(T.transpose(w4),step3_result) + b4 > 0
     step5_result = T.dot(T.transpose(w5),step4_result) + b5 > 0
     step6_result = T.dot(T.transpose(w6),step5_result) + b6 > 0
     step7_result = T.dot(T.transpose(w7),step6_result) + b7 > 0
     result = T.dot(T.transpose(w8),step7_result) + b8


     diff = result - data_sym # Cross-entropy loss function
     cost = (diff ** 2).mean() # no regulization, this will return the mean of the matrix
     gb1, gb2, gb3, gb4, gb5, gb6, gb7, gb8, gw1, gw2, gw3, gw4, gw5, gw6, gw7, gw8 = T.grad(cost, para)

     # Compile

     train = theano.function(
               inputs=[data_sym],
               outputs=[cost],
               updates=((w1, w1 - lr*gw1),(w2, w2 - lr*gw2), \
                    (w3, w3 - lr*gw3), (w4, w4 - lr*gw4), \
                     (w5, w5 - lr*gw5), (w6, w6 - lr*gw6), \
                     (w7, w7 - lr*gw7), (w8, w8 - lr*gw8), \
                    (b1, b1 - lr*gb1), \
                    (b2, b2 - lr*gb2), (b3, b3 - lr*gb3), \
                    (b4, b4 - lr*gb4), (b5, b5 - lr*gb5), \
                    (b6, b6 - lr*gb6), (b7, b7 - lr*gb7), (b8, b8 - lr*gb8)))

     #predict = theano.function(inputs=[data], outputs=result)

     # Train
     cost_record = []
     for i in range(training_steps):
          cost = train(batch_data)
          cost_record.append(cost)
     w_list_output = [w1.get_value(),w2.get_value(),w3.get_value(),w4.get_value(), \
     w5.get_value(),w6.get_value(),w7.get_value(),w8.get_value()]
     b_list_output = [b1.get_value()[:,[0]],b2.get_value()[:,[0]],b3.get_value()[:,[0]],b4.get_value()[:,[0]], \
     b5.get_value()[:,[0]],b6.get_value()[:,[0]],b7.get_value()[:,[0]],b8.get_value()[:,[0]]]
     return cost_record, w_list_output, b_list_output