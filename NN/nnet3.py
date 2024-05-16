from __future__ import absolute_import, division, print_function
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import blackjack as bj


# get the data set
data = open( "data_sets/blackjack.data.3").readlines()
tags = open( "data_sets/blackjack.tags.3").readlines()
data_clean = []
tags_clean = []
#strip whitespace
first = True
for datum in data:
	if first:
		first = False
		continue
	clean_datum = datum[1:datum.index('\n')-1].strip().split(', ')
	clean_datum[0] = int( clean_datum[0] )
	clean_datum[1] = int( clean_datum[1] )
	print( clean_datum )
	data_clean = data_clean + [ clean_datum ]

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

train_data = np.array( data_clean[1:size] )
train_tags = np.array( tags_clean[1:size] )
test_data = np.array( data_clean[size:] )
test_tags = np.array( tags_clean[size:] )

model = keras.Sequential([
    keras.layers.Dense(54, input_shape=(54,)), 
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])


model.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

train_data = np.array(train_data, dtype=float)

model.fit(train_data, train_tags, epochs=50)

test_data = np.array(test_data, dtype=float)
# print(len(train_data))
# print(len(test_tags))

test_loss, test_acc = model.evaluate(test_data, test_tags)

print('Test accuracy:', test_acc)


# save model
# taken from https://machinelearningmastery.com/save-load-keras-deep-learning-models/
model_json = model.to_json()
with open( "models/blackjackmodel.3.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/blackjackmodel.3.h5")
print( "Model saved" )



wins, losses, ties = bj.test_model( "blackjackmodel.3", 100, True, 3, False )
total = wins + losses + ties
win_percentage = (wins/total)*100.0
loss_percentage = (losses/total)*100.0
tie_percentage = (ties/total)*100.0
print( "Percentage won:  " + str( win_percentage ) )
print( "Percentage lost: " + str( loss_percentage ) )
print( "Percentage tied: " + str( tie_percentage ) )





# open serialized model
# taken from https://machinelearningmastery.com/save-load-keras-deep-learning-models/
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
	# if prediction[0][0] > prediction[0][1]:
	# 	print( str(i) + " stay" )
	# else:
	# 	print( str(i) + " hit" )



results = []

for i in range(0,17):
	results = results + [ "" ]
	for j in range(0,9):
		prediction = model.predict( np.array([ [i+5,j+2] ] ) )
		if prediction[0][0] > prediction[0][1]:
			results[i] = results[i] + "s"
		else:
			results[i] = results[i] + "h"

print( "  ", end="" )
with open('policy2.txt', 'w') as file:
    # Header for column numbers
    print("  ", end="", file=file)
    for x in range(len(results[0])):
        print(" " + str((x + 4) % 10), end="", file=file)
    print(file=file)
    
    # Rows with results
    for i in range(len(results)):
        print(i + 5, end="", file=file)
        if i + 5 < 10:
            print("  ", end="", file=file)
        else:
            print(" ", end="", file=file)
        for j in range(len(results[i])):
            print(results[i][j], end=" ", file=file)
        print(file=file)

# for x in range( len(results[0]) ):
# 	print( " " + str( (x+4)%10 ), end="")
# print( )
# for i in range( len(results) ):
# 	print( i+5, end="" )
# 	if i+5 < 10:
# 		print( "  ", end="" )
# 	else:
# 		print( " ", end="" )
# 	for j in range( len(results[i] ) ):
# 		print( results[i][j], end=" ")
# 	print( )

# output_file = 'model_predictions_state_2.txt'

# with open(output_file, 'w') as file:
# 	print('testing_model')
# 	results = []

# 	for i in range(0,17):
# 		results = results + [ "" ]
# 		for j in range(0,9):
# 			prediction = model.predict( np.array([ [i+5,j+2] ] ) )
# 			if prediction[0][0] > prediction[0][1]:
# 				results[i] = results[i] + "s"
# 			else:
# 				results[i] = results[i] + "h"

# 	print( "  ", end="" )

# 	for x in range( len(results[0]) ):
# 		print( " " + str( (x+4)%10 ), end="" , file = file)
# 	print( )
# 	for i in range( len(results) ):
# 		print( i+5, end="" , file = file)
# 		if i+5 < 10:
# 			print( "  ", end="" , file = file)
# 		else:
# 			print( " ", end="" , file = file)
# 		for j in range( len(results[i] ) ):
# 			print( results[i][j], end=" " , file = file)
# 		print( )


# test_loss, test_acc = model.evaluate(test_data, test_tags)


