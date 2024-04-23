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

json_file = open('models/blackjackmodel.2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json( loaded_model_json, custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform} )
model.load_weights( "models/blackjackmodel.2.h5" )
print( "Model loaded from disk" )

print( "testing model" )



output_file = 'model_predictions2.txt'
with open(output_file, 'w' ) as file:
	for i in range(21):
		prediction = model.predict( np.array([ [i,10] ]) , verbose = 0)
		action = "stay" if prediction[0][0] > prediction[0][1] else "hit"
		print(f"{i} {action}", file=file)