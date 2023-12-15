# e ** x = b
# one hot encoding
import numpy as np
import math

b = 5.2

# print(np.log(b))

# print(math.e ** np.log(b)) # b = math.e ** np.log(b)

import math
import numpy as np
layer_op = [4.8, 1.21, 2.385]

# E = 2.7182818
E = math.e

softmax_output = [0.7, 0.1, 0.2]

target_class = 0

target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] + 
         math.log(softmax_output[1]) * target_output[1] +   # 0
         math.log(softmax_output[2]) * target_output[2])    # 0

print(loss)

loss = -(math.log(softmax_output[0]) * target_output[0])

print(loss)

loss = -(math.log(softmax_output[0]))

print(loss)

print(-math.log(0.7)) # 0.3566749439387324 --> high confidence, low loss
print(-math.log(0.5)) # 0.6931471805599453 --> low confidence, high loss


softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = [0, 1, 1]
neg_loss = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
print(neg_loss)
average_loss = np.mean(neg_loss)
print(average_loss)

# but if any number in softmax_outputs is 0, then we have issues with log and mean as they will be inf

