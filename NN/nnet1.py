from __future__ import absolute_import, division, print_function

# tf and keras
import tensorflow as tf
#from tensorflow import keras
import keras
#from keras.models import model_from_json

# helper libs
import numpy as np
import matplotlib.pyplot as plt
import ast
import blackjack as bj

# introduction
print( "TensorFlow Version: " + tf.__version__ )



##############
#preprocessing the level 1 data set
################
# get the data set
# data = open( "data_sets/blackjack2-2-out.data").readlines()
# tags = open( "data_sets/blackjack2-2-out.tags").readlines()
data = open( "data_sets/blackjack1-2-out.data").readlines()
tags = open( "data_sets/blackjack1-2-out.tags").readlines()
data_clean = []
tags_clean = []

#strip whitespaces and such
first = True
i = 0
for datum in data:

	# skip empty line first
	if first:
		first = False
		continue
	clean_datum = datum[:datum.index('\n')].strip()
	clean_datum = ast.literal_eval(clean_datum)
	if type(clean_datum) is int:
		data_clean = data_clean + [ clean_datum ]
	else:
	  data_clean = data_clean + clean_datum 
		


first = True
for tag in tags:
	if first:
		first = False
		continue
	tag = tag[:tag.index('\n')]
	if tag == "h":
		tags_clean = tags_clean + [ 1.0 ]
	else:
		tags_clean = tags_clean + [ 0.0 ]

size = int( len(data)*(0.75) )

num_features = 2  

train_data = np.array( data_clean[1:size] )
print(train_data)
train_tags = np.array( tags_clean[1:size] )
print(train_tags)
test_data = np.array( data_clean[size:] )
print(test_data)
test_tags = np.array( tags_clean[size:] )

################
#train
##############

#The model in this example a dense 2-layer neurel network. 
#The first layer contained 4096 neurons, while the second only had two, for 'hit' or 'stay.' 
#The 'adam' optimizer was used, with a loss of 'sparse_categorical_crossentropy.' 
#Training and testing data was split 50/50 randomly. There are 10 epochs.
model = keras.Sequential()
model.add( keras.layers.Dense(4096, input_dim=2) )
model.add( keras.layers.Dense(2, activation=tf.nn.softmax) )
model.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])
model.fit(train_data, train_tags, epochs=10)
test_loss, test_acc = model.evaluate(test_data, test_tags)
print('Test accuracy:', test_acc)


#The model will output the accuracy of the model, using a random 25% of input as test cases. The model can then be saved via
# save model
# taken from https://machinelearningmastery.com/save-load-keras-deep-learning-models/
model_json = model.to_json()
with open( "models/blackjackmodel.1.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/blackjackmodel.1.h5")
print( "Model saved" )


#To find a heuristic, hand values from 2-21 were tested on the classifier. To do this we need to first deserialize the model from its file
# open serialized model
# taken from https://machinelearningmastery.com/save-load-keras-deep-learning-models/
json_file = open('models/blackjackmodel.1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json( loaded_model_json, custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform} )
model.load_weights( "models/blackjackmodel.1.h5" )
print( "Model loaded from disk" )



print( "testing model" )

for i in range(21):
	prediction = model.predict( np.array([ [i,10] ]) )
	if prediction[0][0] > prediction[0][1]:
		print( str(i) + " stay" )
	else:
		print( str(i) + " hit" )
		

wins, losses, ties = bj.test_model( "blackjackmodel.1", 1000, True, 1, False )
total = wins + losses + ties
win_percentage = (wins/total)*100.0
loss_percentage = (losses/total)*100.0
tie_percentage = (ties/total)*100.0
print( "Percentage won:  " + str( win_percentage ) )