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

json_file = open('models/blackjackmodel.2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json( loaded_model_json, custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform} )
model.load_weights( "models/blackjackmodel.2.h5" )
print( "Model loaded from disk" )

print( "testing model" )

results = []

for i in range(0,22):
	results = results + [ "" ]
	for j in range(0,22):
		prediction = model.predict( np.array([ [i,j] ] ) )
		if prediction[0][0] > prediction[0][1]:
			results[i] = results[i] + "s"
		else:
			results[i] = results[i] + "h"

print( "  ", end="" )
with open('policy2.txt', 'w') as file:
    # Header for column numbers
    print("  ", end="", file=file)
    for x in range(len(results[0])):
        print(" " + str((x) % 10), end="", file=file)
    print(file=file)
    
    # Rows with results
    for i in range(len(results)):
        print(i , end="", file=file)
        if i  < 10:
            print("  ", end="", file=file)
        else:
            print(" ", end="", file=file)
        for j in range(len(results[i])):
            print(results[i][j], end=" ", file=file)
        print(file=file)

print(results)