from __future__ import absolute_import, division, print_function
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import blackjack as bj

#data set will now include the dealer's upward facing card.
#So where previously we used the single integer for data, we will now use a tuple of players_hand, and dealers_hand respectively.
# ex: [ 18, 13, s ]
#The data set for this consists of 5,323 entries, located in data_sets/blackjack.data.2 and data_sets/blackjack.tags.2. Loading the data is done the same as in model 1.

# get the data set
data = open( "data_sets/blackjack2-2-out.data").readlines()
tags = open( "data_sets/blackjack2-2-out.tags").readlines()
data_clean = []
tags_clean = []
#strip whitespace
first = True
i = 0
for datum in data:

	# skip empty line first
	if first:
		first = False
		continue
	clean_datum = datum[:datum.index('\n')].strip()
	clean_datum = clean_datum[1:-1].split(',')
	clean_datum[0] = int( clean_datum[0] )
	clean_datum[1] = int( clean_datum[1][1:] )
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

#The neural network used a similar layer scheme as the previous, with an 16-neuron second layer. The optimizer was 'nadam,' and there were 100 epochs.

model = keras.Sequential()
model.add( keras.layers.Dense(16, input_dim=2) )
model.add( keras.layers.Dense(2, activation=tf.nn.softmax) )
model.compile(optimizer='nadam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])
model.fit(train_data, train_tags, epochs=100)
test_loss, test_acc = model.evaluate(test_data, test_tags)
print('Test accuracy:', test_acc)



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
for x in range( len(results[0]) ):
	print( " " + str( (x+4)%10 ), end="" )
print( )
for i in range( len(results) ):
	print( i+5, end="" )
	if i+5 < 10:
		print( "  ", end="" )
	else:
		print( " ", end="" )
	for j in range( len(results[i] ) ):
		print( results[i][j], end=" " )
	print( )
	

	
print('testing model')
wins, losses, ties = bj.test_model( "blackjackmodel.2", 1000, True, 2, False )
total = wins + losses + ties
win_percentage = (wins/total)*100.0
loss_percentage = (losses/total)*100.0
tie_percentage = (ties/total)*100.0
print( "Percentage won:  " + str( win_percentage ) )
print( "Percentage lost: " + str( loss_percentage ) )
print( "Percentage tied: " + str( tie_percentage ) )