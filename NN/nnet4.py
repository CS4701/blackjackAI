from __future__ import absolute_import, division, print_function
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import blackjack as bj

print( "TensorFlow Version: " + tf.__version__ )



def filter_lines(data, tag, threshold = 54 - 20):
	filtered_data = []
	filtered_tag = []
	removed_indices = []
	for index, line in enumerate(data):
		num_ones = line[1:].count('1')
		if num_ones > threshold:
			removed_indices.append(index)
		else: 
			filtered_data.append(line)
			filtered_tag.append(tag[index])
	return filtered_data, filtered_tag
		



# get the data set
data = open( "data_sets/blackjack.data.3").readlines()
tags = open( "data_sets/blackjack.tags.3").readlines()
data_clean = []
tags_clean = []

# print(data)
# print(tags)
#eliminate card counting bias
data, tags = filter_lines(data, tags, 12)

# print(data)
# print(tags)



#strip whitespace

first = True
for datum in data:
	if first:
		first = False
		continue
	clean_datum = datum[1:datum.index('\n')-1].strip().split(', ')
	clean_datum[0] = int( clean_datum[0] )
	clean_datum[1] = int( clean_datum[1] )
	# print( clean_datum )
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
    keras.layers.Dense(54, input_shape=(54,)),  # player hand, dealer hand, 52 types of cards
    keras.layers.Dense(64, input_dim = 26, activation='relu'),
    keras.layers.Dense(128, input_dim = 13, activation='relu'),
		keras.layers.Dense(256, activation='sigmoid'),
    keras.layers.Dense(2, activation='softmax')
])
# model = keras.Sequential()
# model.add( keras.layers.Dense( 54, input_dim=54 ) )
# model.add( keras.layers.Dense( 64, input_dim=26 ) )
# model.add( keras.layers.Dense( 128, input_dim=13 ) )
# model.add( keras.layers.Dense(2, activation=tf.nn.softmax) )

model.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

train_data = np.array(train_data, dtype=float)

model.fit(train_data, train_tags, epochs=150)

test_data = np.array(test_data, dtype=float)
# print(len(train_data))
# print(len(test_tags))

test_loss, test_acc = model.evaluate(test_data, test_tags)

# print('Test accuracy:', test_acc)


# save model
# taken from https://machinelearningmastery.com/save-load-keras-deep-learning-models/
model_json = model.to_json()
with open( "models/blackjackmodel.4.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/blackjackmodel.4.h5")
print( "Model saved" )


print('Test accuracy: \n', test_acc)


wins, losses, ties = bj.test_model( "blackjackmodel.4", 1000, True, 3, False )
total = wins + losses + ties
win_percentage = (wins/total)*100.0
loss_percentage = (losses/total)*100.0
tie_percentage = (ties/total)*100.0
print( "Percentage won:  " + str( win_percentage ) )
print( "Percentage lost: " + str( loss_percentage ) )
print( "Percentage tied: " + str( tie_percentage ) )



