from __future__ import absolute_import, division, print_function

# tf and keras
import tensorflow as tf
#from tensorflow import keras
import keras
#from keras.models import model_from_json

# helper libs
import numpy as np
import matplotlib.pyplot as plt
import blackjack as bj
from itertools import product, combinations
import time
import pickle
import numpy as np
import sys

#load model 
#input array to predict

json_file = open('models/blackjackmodel.3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json( loaded_model_json, custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform} )
model.load_weights( "models/blackjackmodel.3.h5" )
print( "Model loaded from disk" )

test = [20, 10] + [1]*12 + [0]* 30 + [1]*10 #test is size 54

output_file = 'model_card_count.txt'
with open(output_file, 'w' ) as file:
	prediction = model.predict( np.array([ test]) , verbose = 0)
	action = "stay" if prediction[0][0] > prediction[0][1] else "hit"
	print(f"{action}", file=file)



