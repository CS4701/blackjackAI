import pygame as pygame
from blackjack_deck_moh import *
from constants import *
import sys
import time
import os

# tf and keras
import tensorflow as tf
#from tensorflow import keras
import keras
#from keras.models import model_from_json

# helper libs
import matplotlib.pyplot as plt
from itertools import product, combinations
import pickle
import numpy as np
import random

#Load Card Counting NN model
json_file = open('../NN/models/blackjackmodel.5.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json( loaded_model_json, custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform} )
model.load_weights( "../NN/models/blackjackmodel.5.h5" )
print( "Model loaded from disk" )

NN_YES = True

pygame.init()

clock = pygame.time.Clock()

gameDisplay = pygame.display.set_mode((display_width, display_height))

pygame.display.set_caption('BlackJack')
gameDisplay.fill(background_color)
pygame.draw.rect(gameDisplay, grey, pygame.Rect(0, 0, 250, 700))

###text object render
def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()

def end_text_objects(text, font, color):
    textSurface = font.render(text, True, color)
    return textSurface, textSurface.get_rect()


#game text display
def game_texts(text, x, y, ai=False):
    TextSurf, TextRect = text_objects(text, textfont)
    TextRect.center = (x, y)
    gameDisplay.blit(TextSurf, TextRect)

    pygame.display.update()
    if ai:
        pygame.time.delay(2000)  # Add a delay for the specified time

        # Clear the text after the delay
        pygame.draw.rect(gameDisplay,background_color , TextRect)  # Draw a black rectangle over the text
        pygame.display.update()


 
def game_finish(text, x, y, color):
    TextSurf, TextRect = end_text_objects(text, game_end, color)
    TextRect.center = (x, y)
    gameDisplay.blit(TextSurf, TextRect)
    pygame.display.update()

def black_jack(text, x, y, color):
    TextSurf, TextRect = end_text_objects(text, blackjack, color)
    TextRect.center = (x, y)
    gameDisplay.blit(TextSurf, TextRect)
    pygame.display.update()
    
#button display
def button(msg, x, y, w, h, ic, ac, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        pygame.draw.rect(gameDisplay, ac, (x, y, w, h))
        if click[0] == 1 != None:
            action()
    else:
        pygame.draw.rect(gameDisplay, ic, (x, y, w, h))

    TextSurf, TextRect = text_objects(msg, font)
    TextRect.center = ((x + (w/2)), (y + (h/2)))
    gameDisplay.blit(TextSurf, TextRect)


def card_to_index(card_value, suit):
    #SUITS order: S, H, C, D
    if card_value in 'J':
        card_value = 11
    elif (card_value == 'A'):
        card_value = 1
    elif card_value == 'Q':
        card_value = 12
    elif card_value == 'K':
        card_value = 13
    else:
        card_value = int(card_value)

    if suit == 'S':
        suit = 4
    elif (suit == 'H'):
        suit = 3
    elif (suit == 'C'):
        suit = 2
    else:
        suit = 1

    index = (card_value - 1) * 4 + suit -1
    return index + 2

 
    
    

class Params():
  def __init__(self, action_type = "random_policy", filename =  None, state_mapping = 1):
  # 'input', 'random_policy', 'fixed_policy'
    self.action_type = action_type
  # Only used for 'random_policy' or 'fixed_policy' input
    self.num_games = 20000

    if not filename:
        return
    # Get the parent directory of the current working directory
    parent_directory = os.path.dirname(os.getcwd())

    # Combine parent directory with the relative path to the policy file
    policy_file_path = os.path.join(parent_directory, filename)
  # Filepath to fixed policy file (only used for 'fixed_policy' input)
    self.fixed_policy_filepath = policy_file_path
    # self.fixed_policy_filepath = os.path.join(os.getcwd(), 'QLearning_policy_mapping_2.policy') 
  # Which state mapping algorithm to use (1 or 2)
    self.state_mapping = state_mapping
    return
  
class Play:


    def __init__(self, params = None):
        self.deck = Deck()
        self.dealer = Hand("Dealer")
        self.player1 = Hand()
        self.players = [self.dealer, self.player1]
        self.turn = 1
        self.deck.shuffle()
        self.params = params



        self.NN_YES = NN_YES
        self.AI_1_HAND = [0]*54 #for Card Counting NN input
        self.AI_2_HAND = [0]*54
        self.AI_3_HAND = [0]*54
        self.NUM_ROUNDS = 1


        if self.NN_YES:
            self.player4 = Hand("AI 3", params = None)
            self.players.append(self.player4)

        if params:
            for i, param in enumerate(params):
                if i == 0:
                    self.player2 = Hand("AI 1", params=param)

                    self.player2.policy = self.load_policy(self.player2)
                    self.players.append(self.player2)
            

                elif i == 1:
                    self.player3 = Hand("AI 2", params=param)

                    self.player3.policy = self.load_policy(self.player3)
                    self.players.append(self.player3)




        

    def blackjack(self): #determine winners
        # self.dealer.value = 0
        # self.dealer.calc_hand()

        for player in self.players:
            player.value = 0

        winners = []
        for player in self.players: #dealer and user
            player.calc_hand()
            if player.value == 21:
                winners.append(player)

        if len(winners) == 0:
            return
        
        self.dealer.display_cards()
        show_dealer_card = pygame.image.load('img/' + self.dealer.card_img[1] + '.png').convert()
        gameDisplay.blit(show_dealer_card, (550, 200))


        if len(winners) > 1:
            names_with_blackjack = " and ".join([winner.name for winner in winners])
            print(names_with_blackjack, "with BlackJack!")
            black_jack(f"It's a Tie! {names_with_blackjack} with BlackJack!", 500, 250, grey)
        else:
            winner = winners[0]
            winner.wins += 1
            print(f'{winner.name} got BlackJack!')
            black_jack( f'{winner.name} got BlackJack!', 500, 250, green)
        
        time.sleep(4)
        self.play_or_exit()  
        
    
    # Load policy from input .policy file into self.policy
    def load_policy(self, player):
        # Policy not needed if a user is playing or a random policy is being used
        if player.action_type in ['random_policy', 'input']:
            return None
        # Read policy file and extract policy
        f = open(player.fixed_policy_filepath, 'r')
        data = f.read()
        data = data.split()
        policy = [int(x) for x in data]
        return policy
    
    def hand_to_state(self, player):
        for player in self.players:
            if player is not self.player4:
                player.value = 0

        for player in self.players:
            player.calc_hand()

        # if player.state_mapping == 1:
        #  return self.sum_hand(player_hand) - 1
        if player is not self.player4:
            if player.state_mapping == 2:
                return (self.player2.value- 1) + (18 * (self.dealer.get_dealer_card() - 1))
            elif player.state_mapping == 3:
                if self.player.usable_ace() and len(self.player.cards) <= 2:
                    return 181 + (self.player.value - 11) + (9 * (self.dealer.get_dealer_card() - 1))
                else:
                    return (self.player.value - 1) + (18 * (self.dealer.get_dealer_card() - 1))
        
    
    def policy_run(self, player):
        if not self.dealer.cards:
           return

        if player is not self.player4:
            state = self.hand_to_state(player)
            action = player.policy[state]

            if action: # hit: add a card to players hand and return
                print(player.name, 'Hitting')
                self.hit(player)
            else: # stick: play out the dealers hand, and score
                print(player.name, 'standing')
                self.stand(player)

        elif (player is self.player4):
            prediction = model.predict( np.array([self.AI_3_HAND]) , verbose = 0)
            action = 0 if prediction[0][0] > prediction[0][1] else 1
            if action: # hit: add a card to players hand and return
                print(player.name, 'Hitting')
                self.hit(player)
            else: # stick: play out the dealers hand, and score
                print(player.name, 'standing')
                self.stand(player)




    def deal(self):
        if self.dealer.cards:
            return
        
        for i in range(2):
            for player in self.players:
                player.add_card(self.deck.deal())

        for player in self.players:
            for card in player.cards:
                card_value, suit  = card
                index = card_to_index(card_value, suit)
                self.AI_1_HAND[index] = 1
                self.AI_2_HAND[index] = 1
                self.AI_3_HAND[index] = 1

        self.player1.calc_hand()
        self.dealer.calc_hand()
        self.AI_1_HAND[0] = self.player1.value
        self.AI_2_HAND[0] = self.player1.value
        self.AI_3_HAND[0] = self.player1.value

        self.AI_1_HAND[1] = self.dealer.value
        self.AI_2_HAND[1] = self.dealer.value
        self.AI_3_HAND[1] = self.dealer.value
        
            

        print("AI 1's cards are:", self.player2.cards)
        print("AI 2's cards are:", self.player3.cards)
        print("AI 3's cards are:", self.player4.cards)
        print(self.AI_3_HAND)


        self.dealer.display_cards()
        self.player1.display_cards()
        self.player_card = 1
        dealer_card = pygame.image.load('img/' + self.dealer.card_img[0] + '.png').convert()
        dealer_card_2 = pygame.image.load('img/back.png').convert()
            
        player_card = pygame.image.load('img/' + self.player1.card_img[0] + '.png').convert()
        player_card_2 = pygame.image.load('img/' + self.player1.card_img[1] + '.png').convert()

        
        game_texts("Dealer's hand is:", 500, 150)

        gameDisplay.blit(dealer_card, (400, 200))
        gameDisplay.blit(dealer_card_2, (550, 200))

        game_texts("Your's hand is:", 500, 400)
        
        gameDisplay.blit(player_card, (300, 450))
        gameDisplay.blit(player_card_2, (410, 450))

        pygame.display.update()

        time.sleep(1)
        self.blackjack()

    def hit(self, player=None):
        if not player:
            player = self.player1
        player.add_card(self.deck.deal())
    

        for card in player.cards:
            card_value, suit  = card
            index = card_to_index(card_value, suit)
            self.AI_1_HAND[index] = 1
            self.AI_2_HAND[index] = 1
            self.AI_3_HAND[index] = 1

        # self.blackjack(player)
        player.value = 0
        if player == self.player1:
            self.player_card += 1
        
            if self.player_card == 2:
                player.calc_hand()

                self.AI_1_HAND[0] = self.player1.value
                self.AI_2_HAND[0] = self.player1.value
                self.AI_3_HAND[0] = self.player1.value

                player.display_cards()
                player_card_3 = pygame.image.load('img/' + self.player1.card_img[2] + '.png').convert()
                gameDisplay.blit(player_card_3, (520, 450))

            if self.player_card == 3:
                player.calc_hand()

                self.AI_1_HAND[0] = self.player1.value
                self.AI_2_HAND[0] = self.player1.value
                self.AI_3_HAND[0] = self.player1.value

                player.display_cards()
                player_card_4 = pygame.image.load('img/' + self.player1.card_img[3] + '.png').convert()
                gameDisplay.blit(player_card_4, (630, 450))
            
            if self.player_card == 4:
                player.calc_hand()

                self.AI_1_HAND[0] = self.player1.value
                self.AI_2_HAND[0] = self.player1.value
                self.AI_3_HAND[0] = self.player1.value

                player.display_cards()
                player_card_5 = pygame.image.load('img/' + self.player1.card_img[4] + '.png').convert()
                gameDisplay.blit(player_card_5, (740, 450))
            
            pygame.display.update()

        else:
            player.calc_hand()

            self.AI_1_HAND[0] = self.player1.value
            self.AI_2_HAND[0] = self.player1.value
            self.AI_3_HAND[0] = self.player1.value
            
            print(player.name, "cards:", player.cards)
            time.sleep(1)
            
        if player.value == 21:
            self.turn += 1

        if player.value > 21:
            player.busted = True
            print(player.name, "BUSTED!")
            print(player.cards)
            self.turn += 1
            time.sleep(2)
            
        player.value = 0

            
    def stand(self, player=None):
        if not player: #not dealer
            player = self.player1
        self.turn += 1  
        time.sleep(1)      
        

    def check_winner(self):
        show_dealer_card = pygame.image.load('img/' + self.dealer.card_img[1] + '.png').convert()
        gameDisplay.blit(show_dealer_card, (550, 200))


        for player in self.players:
            player.value = 0
            player.calc_hand()
            if player.value == 21:
                self.blackjack()
                return

        for player in self.players: #print dealer and user total
            print(player.name, 'total:', player.value)
    
        highest_score = max([player.value for player in self.players if not player.busted], default=0)
        winners = [player for player in self.players if player.value == highest_score and not player.busted]
        print()
        if len(winners) == 1:
            winner = winners[0]
            print(f'{winner.name} Won!')
            winner.wins += 1
            game_finish(f'{winner.name} Wins!', 500, 250, green)
        else:
            winner_names = " and ".join([winner.name for winner in winners])
            print(f'It\'s a Tie! {winner_names} Won!')
            game_finish(f'It\'s a Tie! {winner_names} Won!', 500, 250, grey)

        print()
        time.sleep(4)
        self.play_or_exit()
      
        return
    
    def exit(self):
        sys.exit()
    
    def play_or_exit(self):
        if self.NUM_ROUNDS == 3:
            self.NUM_ROUNDS = 0
            self.AI_1_HAND = [0]*54 
            self.AI_2_HAND = [0]*54
            self.AI_3_HAND = [0]*54
        self.NUM_ROUNDS +=1

        game_texts("Play again press Deal!", 200, 80)
        self.turn = 1

        if self.NUM_ROUNDS == 1:
            self.deck = Deck()
        print("THE ROUND IS: ", self.NUM_ROUNDS)

        for player in self.players:
            player.reset()
            print(player.name, "wins:", player.wins)
        print()
        self.deck.shuffle()
        gameDisplay.fill(background_color)
        pygame.draw.rect(gameDisplay, grey, pygame.Rect(0, 0, 250, 700))
        pygame.display.update()
    
    

params = [Params('fixed_policy', 'QLearning_policy_mapping_2.policy', 2), Params('fixed_policy', 'QLearning_policy_mapping_2.policy', 2) ]   
play_blackjack = Play(params)

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        button("Deal", 30, 100, 150, 50, light_slat, dark_slat, play_blackjack.deal)
        button("Hit", 30, 200, 150, 50, light_slat, dark_slat, play_blackjack.hit)
        button("Stand", 30, 300, 150, 50, light_slat, dark_slat, play_blackjack.stand)
        button("EXIT", 30, 500, 150, 50, light_slat, dark_red, play_blackjack.exit)

        if play_blackjack.turn == 2:
            play_blackjack.policy_run(play_blackjack.player2)
            print('player 2')
            time.sleep(1)
        if play_blackjack.turn == 3:
            play_blackjack.policy_run(play_blackjack.player3)
            print('player 3')
            time.sleep(1)

        if NN_YES:
            if play_blackjack.turn == 4:
                play_blackjack.policy_run(play_blackjack.player4)
            if play_blackjack.turn == 5:
                print()
                print('checking winner')
                play_blackjack.check_winner()
        else: 
            if play_blackjack.turn == 4:
                print()
                print('checking winner')
                play_blackjack.check_winner()
    
    pygame.display.flip()