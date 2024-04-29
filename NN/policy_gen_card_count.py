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
import random

#load model 
#input array to predict

#First load the Card Counting NN model
json_file = open('models/blackjackmodel.3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json( loaded_model_json, custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform} )
model.load_weights( "models/blackjackmodel.3.h5" )
print( "Model loaded from disk" )



#Construct input for the NN 
# NOTE: when anyone bust, (meaning sum > 21) we don't use the model to make anymore more prediction.
#Keep track of player_hand_sum when dealer first deal the cards to player/ continue to update player_hand_sum until player bust or stand
player_hand_sum =  random.randint(2, 21) #number 2 to 21

#Keep track of dealer_hand_sum when dealer bust or stand
dealer_hand_sum = random.randint(2, 21)

#initialize a vector to check the cards that is played/dealt
input_vector = [player_hand_sum, dealer_hand_sum] + [0]*52


prediction = model.predict( np.array([input_vector]) , verbose = 0)
action = "stay" if prediction[0][0] > prediction[0][1] else "hit"
print(action)





