import pygame as pygame
from blackjack_deck_moh import *
from constants import *
import sys
import time
import os
import tensorflow as tf
from keras import models
import matplotlib.pyplot as plt
from itertools import product, combinations
import pickle
import numpy as np
import random

# Load Card Counting NN model
json_file = open('../NN/models/blackjackmodel.5.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = models.model_from_json( loaded_model_json, custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform} )
model.load_weights( "../NN/models/blackjackmodel.5.h5" )
print( "Model loaded from disk" )


NN_YES = True
ai1_policy = None
ai2_policy = None
ai1_statemapping = 1
ai2_statemapping = 2
game_winners = [0, 0, 0, 0]
total_games = [0,0,0,0]
run_count = 0

pygame.init()

clock = pygame.time.Clock()

gameDisplay = pygame.display.set_mode((display_width, display_height))
CARD_WIDTH = pygame.image.load('img/2C.png').convert().get_width()*2/3  
CARD_HEIGHT = pygame.image.load('img/2C.png').convert().get_height()*2/3  
CARD_SPACING = 20 
pygame.display.set_caption('BlackJack')
gameDisplay.fill(background_color)
# pygame.draw.rect(gameDisplay, grey, pygame.Rect(0, 0, 250, 700))

###text object render
def text_objects(text, font):
    font = pygame.font.Font(None, 16)
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()

def end_text_objects(text, font, color):
    font = pygame.font.Font(None, 32)
    textSurface = font.render(text, True, color)
    return textSurface, textSurface.get_rect()


#game text display
def game_texts(text, x, y, ai=False):
    TextSurf, TextRect = text_objects(text, textfont)
    TextRect.center = (x, y)
    gameDisplay.blit(TextSurf, TextRect)

    pygame.display.update()
    if ai:
        pygame.time.delay(3000)  # Add a delay for the specified time

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
def button(msg, x, y, w, h, ic, ac, action=None, active=True, params = None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        pygame.draw.rect(gameDisplay, ac if active else ic, (x, y, w, h))
        if click[0] == 1 and action and active:
            if params is not None:
                action(*params)
            else:
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
        self.game_active = False
        self.deck = Deck()
        self.dealer = Hand("Dealer")
        self.player1 = Hand()
        self.players = [self.dealer, self.player1]
      

        self.turn = 1
        self.deck.shuffle()
        self.params = params




        self.AI_1_HAND = [0]*54 #for Card Counting NN input
        self.AI_2_HAND = [0]*54
        # self.AI_3_HAND = [0]*54
        self.NUM_ROUNDS = 1


        if params:
            for i, param in enumerate(params):

                if param is not None and param.fixed_policy_filepath:
                    if i == 0:
                        self.player2 = Hand("AI 1", NN = False, params=param)

                        self.player2.policy = self.load_policy(self.player2)
                        self.players.append(self.player2)
                        
                

                    elif i == 1:
                        self.player3 = Hand("AI 2", NN = False, params=param)

                        self.player3.policy = self.load_policy(self.player3)
                        self.players.append(self.player3)

                elif param is None:
                    if i == 0:
                        self.player2 = Hand("AI 1", NN = True, params=None)
                        self.players.append(self.player2)
                

                    elif i == 1:
                        self.player3 = Hand("AI 2", NN = True, params=None)
                        self.players.append(self.player3)


                else:
                    print("No file path")


    def clear_card_areas(self, start_x, start_y, width):
        gameDisplay.fill(background_color, rect=[start_x, start_y, width, CARD_HEIGHT])
        pygame.display.update()

    def draw_cards(self, card_images, player, show_dealer=False):
            num_cards = len(card_images)
            start_x = 0
            start_y = 0
            total_width = num_cards * CARD_WIDTH + (num_cards - 1) * CARD_SPACING
            if player == 0 or player == 2:
                start_x = (display_width - total_width) / 2  # Start so that cards are centered
                if player == 0:
                    start_y = 80

                else:
                    start_y = 500

                self.clear_card_areas(start_x, start_y, total_width)
            elif player == 1 or player == 3:
                start_y = (display_height)/2
                if player == 1:
                    start_x = display_width/4 - total_width/2
                if player == 3:
                    start_x = 3 * display_width/4 - total_width/2
                self.clear_card_areas(start_x, start_y, total_width)


            if player == 0 and show_dealer == False:
                for i in range(len(card_images)):
                        card_image = pygame.image.load('img/' + card_images[i] + '.png').convert()
                        card_image = pygame.transform.scale(card_image, (CARD_WIDTH, CARD_HEIGHT))
                        gameDisplay.blit(card_image, (start_x + i * (CARD_WIDTH + CARD_SPACING), start_y))
                card_image = pygame.image.load('img/back.png').convert()
                card_image = pygame.transform.scale(card_image, (CARD_WIDTH, CARD_HEIGHT))
                gameDisplay.blit(card_image, (start_x + i * (CARD_WIDTH + CARD_SPACING), start_y))
            else:
                for i in range(len(card_images)):
                        card_image = pygame.image.load('img/' + card_images[i] + '.png').convert()
                        card_image = pygame.transform.scale(card_image, (CARD_WIDTH, CARD_HEIGHT))
                        gameDisplay.blit(card_image, (start_x + i * (CARD_WIDTH + CARD_SPACING), start_y))
            pygame.display.update()

        

    def run_count_update(self, card):
        global run_count
        if card[0] in 'JQKA':
            run_count -= 1
        elif int(card[0]) < 7:
            run_count += 1


    def blackjack(self): #determine winners
        # self.dealer.value = 0
        # self.dealer.calc_hand()
        global game_winners
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
        # show_dealer_card = pygame.image.load('img/' + self.dealer.card_img[1] + '.png').convert()
        # gameDisplay.blit(show_dealer_card, (550, 200))
        self.draw_cards(self.dealer.card_img, 0, True)

        

        if len(winners) > 1 and self.dealer in winners:
            names_with_blackjack = " and ".join([winner.name for winner in winners])
            print(names_with_blackjack, "with BlackJack!")
            black_jack(f"It's a Tie! {names_with_blackjack} with BlackJack!", display_width/2, 250, grey)
            for winner in winners:
                if winner == self.dealer:
                    game_winners[0] += 1
                elif winner == self.player1:
                    game_winners[1] += 1
                elif winner == self.player2:
                    game_winners[2] += 1
                elif winner == self.player3:
                    game_winners[3] += 1
                else:
                    print("no such player")

        elif len(winners) > 1:
            winner_names = " and ".join([winner.name for winner in winners])

            # for w in winners:
            #     w.wins +=1
            #     print(f'{w.name} got BlackJack!')
            black_jack( f'{winner_names} got BlackJack!', display_width/2, 250, green)
            for winner in winners:
                if winner == self.dealer:
                    game_winners[0] += 1
                elif winner == self.player1:
                    game_winners[1] += 1
                elif winner == self.player2:
                    game_winners[2] += 1
                elif winner == self.player3:
                    game_winners[3] += 1
                else:
                    print("no such player")
        else:
            winner = winners[0]
            # winner.wins += 1
            if winner == self.dealer:
                game_winners[0] += 1
            elif winner == self.player1:
                game_winners[1] += 1
            elif winner == self.player2:
                game_winners[2] += 1
            elif winner == self.player3:
                game_winners[3] += 1
            else:
                print("no such player")
            print(f'{winner.name} got BlackJack!')
            black_jack( f'{winner.name} got BlackJack!', display_width/2, 250, green)
        
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
    
        # testing if policy failed v
        # if hasattr(player, 'fixed_policy_filepath') and player.fixed_policy_filepath:
        #     with open(player.fixed_policy_filepath, 'r') as f:
        #         policy_data = f.read()
        #     # Load the policy from the data here
        #     return policy_data
        # else:
        #     raise ValueError("Policy file path not set for the player.")
    
    def hand_to_state(self, player):
        global run_count
        # print(player.state_mapping)
        for p in self.players:
            p.value = 0

        for p in self.players:
            p.calc_hand()
        
        # print(player.state_mapping)

        if player.state_mapping == 1:
            # print(f'state_mapping_1 {player.value}')
            return player.value - 1
        elif player.state_mapping == 2:
            # print(f'state_mapping_2 {player.value}')
            return (player.value- 1) + (18 * (self.dealer.get_dealer_card() - 1))
        elif player.state_mapping == 3:
            if player.useable_ace() and len(player.cards) <= 2:
                return 181 + (player.value - 11) + (9 * (self.dealer.get_dealer_card() - 1))
            else:
                return (player.value - 1) + (18 * (self.dealer.get_dealer_card() - 1))
        elif player.state_mapping == 4:

            return player.value + self.dealer.get_dealer_card() * 21 + (run_count + 10) * 21 * 21
    
    def policy_run(self, player):
        if not self.dealer.cards:
           return
        if player.NN == False:
            if (player is self.player2) or (player is self.player3):
                # print(player.name)
                # print(player.state_mapping)
                state = self.hand_to_state(player)
                # print(f"state: {state}")
                action = player.policy[state]

                if action: # hit: add a card to players hand and return
                    print(player.name, 'Hitting')
                    self.hit(player)
                else: # stick: play out the dealers hand, and score
                    print(player.name, 'standing')
                    self.stand(player)

        elif (player.NN):
            if self.turn == 2:
                prediction = model.predict( np.array([self.AI_1_HAND]) , verbose = 0)
                action = 0 if prediction[0][0] > prediction[0][1] else 1
                if action: # hit: add a card to players hand and return
                    print(player.name, 'Hitting')
                    self.hit(player)
                else: # stick: play out the dealers hand, and score
                    print(player.name, 'standing')
                    self.stand(player)
            elif self.turn == 3:
                prediction = model.predict( np.array([self.AI_2_HAND]) , verbose = 0)
                action = 0 if prediction[0][0] > prediction[0][1] else 1
                if action: # hit: add a card to players hand and return
                    print(player.name, 'Hitting')
                    self.hit(player)
                else: # stick: play out the dealers hand, and score
                    print(player.name, 'standing')
                    self.stand(player)
            else:
                print('There is a bug my friend')





    def deal(self):
        global total_games
        print('===========================================')
        for i in range(len(total_games)):
            total_games[i]+=1

        if not self.game_active:
            self.game_active = True
        if self.dealer.cards:
            return
        
        for _ in range(2):
            for player in self.players:
                player.add_card(self.deck.deal())
                self.run_count_update(player.cards[-1])

        for player in self.players:
            for card in player.cards:
                card_value, suit  = card
                index = card_to_index(card_value, suit)
                self.AI_1_HAND[index] = 1
                self.AI_2_HAND[index] = 1

        self.player1.calc_hand()
        self.dealer.calc_hand()
        self.AI_1_HAND[0] = self.player1.value
        self.AI_2_HAND[0] = self.player1.value

        self.AI_1_HAND[1] = self.dealer.value
        self.AI_2_HAND[1] = self.dealer.value
        
            

        print("AI 1's cards are:", self.player2.cards)
        print("AI 2's cards are:", self.player3.cards)
        # print("AI 3's cards are:", self.player4.cards)
        # print(self.AI_3_HAND)


        self.dealer.display_cards()
        self.player1.display_cards()
        self.player2.display_cards()
        self.player3.display_cards()
        self.player_card = 1

        # dealer_card = pygame.image.load('img/' + self.dealer.card_img[0] + '.png').convert()
        # dealer_card_2 = pygame.image.load('img/back.png').convert()
            
        # player_card = pygame.image.load('img/' + self.player1.card_img[0] + '.png').convert()
        # player_card_2 = pygame.image.load('img/' + self.player1.card_img[1] + '.png').convert()

        
        game_texts("Dealer's hand is:", display_width/2, 55)
        game_texts("AI 1's hand is:", display_width/4, display_height/2-25)
        game_texts("AI 2's hand is:", 3*display_width/4, display_height/2-25)

        # gameDisplay.blit(dealer_card, (400, 200))
        # gameDisplay.blit(dealer_card_2, (550, 200))
        self.draw_cards(self.dealer.card_img, 0)


        game_texts("Your's hand is:", display_width/2, 475)
        
        # gameDisplay.blit(player_card, (300, 450))
        # gameDisplay.blit(player_card_2, (410, 450))

        # pygame.display.update()
        self.draw_cards(self.player1.card_img, 2)

        self.draw_cards(self.player2.card_img, 1)
        self.draw_cards(self.player3.card_img, 3)


        time.sleep(1)
        self.blackjack()

    def hit(self, player=None):
        if not player:
            player = self.player1
        player.add_card(self.deck.deal())
        self.run_count_update(player.cards[-1])
    

        for card in player.cards:
            card_value, suit  = card
            index = card_to_index(card_value, suit)
            self.AI_1_HAND[index] = 1
            self.AI_2_HAND[index] = 1
            # self.AI_3_HAND[index] = 1

        # self.blackjack(player)
        player.value = 0
        if player == self.player1:
            self.player_card += 1
        
            if self.player_card == 2:
                player.calc_hand()

                self.AI_1_HAND[0] = self.player1.value
                self.AI_2_HAND[0] = self.player1.value
                # self.AI_3_HAND[0] = self.player1.value

                player.display_cards()
                # player_card_3 = pygame.image.load('img/' + self.player1.card_img[2] + '.png').convert()
                # gameDisplay.blit(player_card_3, (520, 450))
                self.draw_cards(self.player1.card_img, 2)


            if self.player_card == 3:
                player.calc_hand()

                self.AI_1_HAND[0] = self.player1.value
                self.AI_2_HAND[0] = self.player1.value
                # self.AI_3_HAND[0] = self.player1.value

                player.display_cards()
                # player_card_4 = pygame.image.load('img/' + self.player1.card_img[3] + '.png').convert()
                # gameDisplay.blit(player_card_4, (630, 450))
                self.draw_cards(self.player1.card_img, 2)

            
            if self.player_card == 4:
                player.calc_hand()

                self.AI_1_HAND[0] = self.player1.value
                self.AI_2_HAND[0] = self.player1.value
                # self.AI_3_HAND[0] = self.player1.value

                player.display_cards()
                # player_card_5 = pygame.image.load('img/' + self.player1.card_img[4] + '.png').convert()
                # gameDisplay.blit(player_card_5, (740, 450))
                self.draw_cards(self.player1.card_img, 2)

            pygame.display.update()

        else:
            player.calc_hand()
            if player == self.player2:
                player.display_cards()
                self.draw_cards(self.player2.card_img, 1)
            if player == self.player3:
                player.display_cards()
                self.draw_cards(self.player3.card_img, 3)


            self.AI_1_HAND[0] = self.player1.value
            self.AI_2_HAND[0] = self.player1.value
            # self.AI_3_HAND[0] = self.player1.value
            
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


    def play_dealer(self):
        self.dealer.value = 0
        self.dealer.calc_hand()

        # show_dealer_card = pygame.image.load('img/' + self.dealer.card_img[1] + '.png').convert()
        # gameDisplay.blit(show_dealer_card, (550, 200))

        while (self.dealer.value < 16):
            print("dealer hitting")
            self.dealer.add_card(self.deck.deal())
            self.run_count_update(self.dealer.cards[-1])
            print("dealer cards", self.dealer.cards)
            self.dealer.display_cards()
            # print("dealer cards", self.dealer.card_img)
            # last_dealer_card = pygame.image.load('img/' + self.dealer.card_img[-1] + '.png').convert()
            # gameDisplay.blit(last_dealer_card, (550 + 110 * (len(self.dealer.cards) - 2), 200))
            self.dealer.value = 0
            self.dealer.calc_hand()

        if self.dealer.value == 21:
            self.turn += 1
        elif self.dealer.value > 21:
            self.dealer.busted = True
            print(self.dealer.name, "BUSTED!")
            print(self.dealer.cards)
            self.turn += 1
            time.sleep(2)
        else:
            self.stand(self.dealer)
            
        self.dealer.value = 0     
        
    def check_winner(self):
        global game_winners
        # show_dealer_card = pygame.image.load('img/' + self.dealer.card_img[1] + '.png').convert()
        # gameDisplay.blit(show_dealer_card, (550, 200))
        self.draw_cards(self.dealer.card_img, 0, True)
        print("My cards are:", self.player1.cards)
        print("Dealer cards are:", self.dealer.cards)
        print("AI1 cards are:", self.player2.cards)
        print("AI2 cards are:", self.player3.cards)

        for player in self.players:
            player.value = 0
            player.calc_hand()
            if player.value == 21:
                self.blackjack()
                return

        for player in self.players: #print dealer and user total
            print(player.name, 'total:', player.value)

        dealer_score = 0
        if not self.dealer.busted:
            dealer_score = self.dealer.value 

        highest_score = max([player.value for player in self.players if not player.busted], default=0)

        if dealer_score == highest_score: #tie game
            winners = [player for player in self.players if player.value >= dealer_score and not player.busted]
        else: #tie game without dealer
            winners = [player for player in self.players if player.value > dealer_score and not player.busted]


        # highest_score = max([player.value for player in self.players if not player.busted], default=0)
        # winners = [player for player in self.players if player.value == highest_score and not player.busted]
        print()
        if len(winners) == 1:
            winner = winners[0]
            print(f'{winner.name} Won!')
            # winner.wins += 1
            if winner == self.dealer:
                game_winners[0] += 1
            elif winner == self.player1:
                game_winners[1] += 1
            elif winner == self.player2:
                game_winners[2] += 1
            elif winner == self.player3:
                game_winners[3] += 1
            else:
                print("no such player")
            game_finish(f'{winner.name} Wins!', 500, 250, green)
       
        elif self.dealer not in winners:
            winner_names = " and ".join([winner.name for winner in winners])
            game_finish(f'{winner_names} Win!', 500, 250, green)

            # for w in winners:
            #     w.wins+=1
            for winner in winners:
                if winner == self.dealer:
                    game_winners[0] += 1
                elif winner == self.player1:
                    game_winners[1] += 1
                elif winner == self.player2:
                    game_winners[2] += 1
                elif winner == self.player3:
                    game_winners[3] += 1
                else:
                    print("no such player")
            
        else:
            winner_names = " and ".join([winner.name for winner in winners])
            print(f'It\'s a Tie! {winner_names} Won!')
            game_finish(f'It\'s a Tie! {winner_names} Won!', 500, 250, grey)
            for winner in winners:
                if winner == self.dealer:
                    game_winners[0] += 1
                elif winner == self.player1:
                    game_winners[1] += 1
                elif winner == self.player2:
                    game_winners[2] += 1
                elif winner == self.player3:
                    game_winners[3] += 1
                else:
                    print("no such player")
        print()
        time.sleep(4)
        self.play_or_exit()
      
        return
    
    def exit(self):
        sys.exit()
    
    def play_or_exit(self):
        global run_count
        print(f'run count: {run_count}')
        self.game_active = False
        if self.NUM_ROUNDS == 3:
            self.NUM_ROUNDS = 0
            self.AI_1_HAND = [0]*54 
            self.AI_2_HAND = [0]*54
            run_count = 0
            # self.AI_3_HAND = [0]*54
        self.NUM_ROUNDS +=1

        game_texts("Play again press Deal!", 200, 80)
        self.turn = 1

        if self.NUM_ROUNDS == 1:
            self.deck = Deck()
        print("THE ROUND IS: ", self.NUM_ROUNDS)

        for player in self.players:
            player.reset()
            # print(player.name, "wins:", player.wins)
        print()
        self.deck.shuffle()
        gameDisplay.fill(background_color)
        # pygame.draw.rect(gameDisplay, grey, pygame.Rect(0, 0, 250, 700))
        pygame.display.update()
    
    



def game():
    global ai1_policy, ai2_policy, ai1_statemapping, ai2_statemapping
    # print(f"{ai1_statemapping}    {ai2_statemapping}")
    if ai1_policy is None and ai2_policy is None:
        ai1_policy = 'QLearning_policy_mapping_1.policy'
        ai2_policy = "QLearning_policy_mapping_2.policy"


    if ai1_policy == 'NN' and ai2_policy == 'NN':
        params = [None, None]
    elif ai2_policy == 'NN':
         params = [Params('fixed_policy', ai1_policy, ai1_statemapping), None]
    elif ai1_policy == 'NN':
         params = [None ,Params('fixed_policy', ai2_policy, ai2_statemapping) ] 
    else: 
        params = [Params('fixed_policy', ai1_policy, ai1_statemapping), Params('fixed_policy', ai2_policy, ai2_statemapping) ]   


    play_blackjack = Play(params)
    running = True
    gameDisplay.fill(background_color)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            button("Deal", 770, 500, 120, 40, light_slat, dark_slat, play_blackjack.deal, not play_blackjack.game_active)
            button("Hit", 770, 550, 120, 40, light_slat, dark_slat, play_blackjack.hit, play_blackjack.game_active)
            button("Stand", 770, 600, 120, 40, light_slat, dark_slat, play_blackjack.stand, play_blackjack.game_active)
            button("EXIT", 770, 650, 120, 40, light_slat, dark_red, play_blackjack.exit)
            button("win rates", 30, 600, 120, 40, light_slat, dark_slat, show_winrate_screen, not play_blackjack.game_active, [play_blackjack])

            button("Change", 30, 650, 120, 40, light_slat, dark_slat, show_policy_selection_screen, not play_blackjack.game_active)



            if play_blackjack.turn == 2:
                play_blackjack.policy_run(play_blackjack.player2)
                print('player 1')
                time.sleep(1)
            if play_blackjack.turn == 3:
                play_blackjack.policy_run(play_blackjack.player3)
                print('player 2')
                time.sleep(1)

            if play_blackjack.turn == 4:
                print('Dealer')
                play_blackjack.play_dealer()
                time.sleep(1)

            if play_blackjack.turn == 5:
                print()
                print('checking winner')
                play_blackjack.check_winner()
        
        pygame.display.flip()


def option_text(text, x, y, color, select_panel= False):
    TextSurf, TextRect = end_text_objects(text, blackjack, color)
    TextRect.center = (x, y)
    gameDisplay.blit(TextSurf, TextRect)
    pygame.display.update()
    Y = TextRect.y
    if select_panel:

        TextRect.y = Y + 400
        pygame.draw.rect(gameDisplay, background_color , TextRect)

        TextRect.y = Y + 300
        pygame.draw.rect(gameDisplay, background_color , TextRect)

        TextRect.y = Y + 200
        pygame.draw.rect(gameDisplay, background_color , TextRect)

        TextRect.y = Y + 100
        pygame.draw.rect(gameDisplay, background_color , TextRect)
        
        TextRect.y = Y - 100
        pygame.draw.rect(gameDisplay,background_color , TextRect)

        TextRect.y = Y - 200
        pygame.draw.rect(gameDisplay,background_color , TextRect)

        TextRect.y = Y - 300
        pygame.draw.rect(gameDisplay, background_color , TextRect)

        TextRect.y = Y - 400
        pygame.draw.rect(gameDisplay, background_color , TextRect)



def set_policy(ai, policy_name, statemapping = None ):
    global ai1_policy, ai2_policy, ai1_statemapping, ai2_statemapping, game_winners
    if ai == 'AI1':
        total_games[2]=0
        game_winners[2]= 0
        ai1_policy = policy_name
        ai1_statemapping = statemapping
        # print(f"state mapping for AI 1: {statemapping}")
    elif ai == 'AI2':
        total_games[3]=0
        game_winners[3]=0
        ai2_policy = policy_name
        ai2_statemapping = statemapping
        # print(f"state mapping for AI 2: {statemapping}")

    gameDisplay.fill(background_color, rect=[225, 200, 500, 50])
    pygame.display.update()
    # game_winners = [0,0,0,0]
    # print(f"{ai} policy set to {policy_name}")
    # print(f"{ai} statemapping set to {statemapping}")


def show_policy_selection_screen():
    running = True
    gameDisplay.fill(background_color)
    option_text("AI 1's policy", 150, 100, black)
    option_text("AI 2's policy", 750, 100, black)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        button("Q Learning 1.0", 100, 200, 100, 40, light_slat, dark_slat, set_policy, True, ["AI1", 'QLearning_policy_mapping_1.policy', 1])
        button("Q Learning 2.0", 700, 200, 100, 40, light_slat, dark_slat, set_policy, True, ["AI2", 'QLearning_policy_mapping_2.policy', 2])

        button("QL CC", 100, 300, 100, 40, light_slat, dark_slat, set_policy, True, ["AI1", "qlearningcc.policy", 4]) 
        button("NN Card Count", 700, 300, 100, 40, light_slat, dark_slat, set_policy, True, ["AI2", "NN"])

        button("Value Iteration 1.0", 100, 400, 100, 40, light_slat, dark_slat, set_policy, True, ["AI1", 'Value_Iteration_Policy_1.policy', 1])
        button("Value Iteration 2.0", 700, 400, 100, 40, light_slat, dark_slat, set_policy, True, ["AI2", 'Value_Iteration_Policy_2.policy', 2])

        button("SARSA 1.0", 100, 500, 100, 40, light_slat, dark_slat, set_policy, True, ["AI1", 'Sarsa_Policy_1.policy', 1])
        button("SARSA 2.0", 700, 500, 100, 40, light_slat, dark_slat, set_policy, True, ["AI2", 'Sarsa_Policy_2.policy', 2])


        button("Math Policy", 100, 600, 100, 40, light_slat, dark_slat, set_policy, True, ["AI1", "math_policy.policy", 3]) 
        button("NN no CC", 700, 600, 100, 40, light_slat, dark_slat, set_policy, True, ["AI2", "model_predictions2.policy", 1])

        button("Back", 770, 650, 120, 40, light_slat, dark_slat, game, True)

        if ai1_policy == 'QLearning_policy_mapping_1.policy':
            option_text('Selected', 325, 225, black, select_panel=True)
        elif ai1_policy == 'qlearningcc.policy':
            option_text('Selected', 325, 325, black, select_panel= True)
        elif ai1_policy == 'Value_Iteration_Policy_1.policy':
            option_text('Selected', 325, 425, black, select_panel= True)
        elif ai1_policy == 'Sarsa_Policy_1.policy':
            option_text('Selected', 325, 525, black, select_panel= True)
        elif ai1_policy == 'math_policy.policy':
            option_text('Selected', 325, 625, black, select_panel= True)



        if ai2_policy == 'QLearning_policy_mapping_2.policy':
            option_text('Selected', 575, 225, black, select_panel= True)
        elif ai2_policy == 'NN':
            option_text('Selected', 575, 325, black, select_panel= True)
        elif ai2_policy == 'Value_Iteration_Policy_2.policy':
            option_text('Selected', 575, 425, black, select_panel= True)
        elif ai2_policy == 'Sarsa_Policy_2.policy':
            option_text('Selected', 575, 525, black, select_panel= True)
        elif ai2_policy == 'model_predictions2.policy':
            option_text('Selected', 575, 625, black, select_panel= True)

        pygame.display.flip()


def show_winrate_screen(game_state):
    global game_winners, total_games

    running = True
    gameDisplay.fill(background_color)
    option_text(f"AI 1's win rate: {game_winners[2]}/{total_games[2]}", 150, 100, black)
    option_text(f"AI 2's win rate: {game_winners[3]}/{total_games[3]}", 150, 200, black)
    option_text(f"Dealer's win rate: {game_winners[0]}/{total_games[0]}", 150, 300, black)
    option_text(f"Your win rate: {game_winners[1]}/{total_games[1]}", 150, 400, black)
    # option_text(f"AI 1's win: {game_state.player2.wins}", 150, 100, black)
    # option_text(f"AI 2's win: {game_state.player3.wins}", 150, 200, black)
    # option_text(f"Dealer's win: {game_state.dealer.wins}", 150, 300, black)
    # option_text(f"Your win: {game_state.player1.wins}", 150, 400, black)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        button("back", 770, 650, 120, 40, light_slat, dark_slat, game, True)

        pygame.display.flip()


game()