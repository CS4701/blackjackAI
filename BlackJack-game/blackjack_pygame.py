import pygame as pygame
from blackjack_deck import *
from constants import *
import sys
import time
import os
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

class Params():
  def __init__(self):
  # 'input', 'random_policy', 'fixed_policy'
    self.action_type = 'fixed_policy'
  # Only used for 'random_policy' or 'fixed_policy' input
    self.num_games = 20000
    # Get the parent directory of the current working directory
    parent_directory = os.path.dirname(os.getcwd())

    # Combine parent directory with the relative path to the policy file
    policy_file_path = os.path.join(parent_directory, 'QLearning_policy_mapping_2.policy')
  # Filepath to fixed policy file (only used for 'fixed_policy' input)
    self.fixed_policy_filepath = policy_file_path
    # self.fixed_policy_filepath = os.path.join(os.getcwd(), 'QLearning_policy_mapping_2.policy') 
  # Which state mapping algorithm to use (1 or 2)
    self.state_mapping = 2
    return
  
class Play:
    def __init__(self, params = None):
        self.deck = Deck()
        self.dealer = Hand()
        self.player = Hand()
        self.deck.shuffle()
        if params:
            self.action_type = params.action_type
            self.fixed_policy_filepath = params.fixed_policy_filepath
            self.policy = self.load_policy()
            self.state_mapping = params.state_mapping

    def clear_card_areas(self, start_x, start_y, width):
        gameDisplay.fill(background_color, rect=[start_x, start_y, width, CARD_HEIGHT])
        pygame.display.update()

    def draw_cards(self, card_images, player, show_dealer=False):
            num_cards = len(card_images)
            total_width = num_cards * CARD_WIDTH + (num_cards - 1) * CARD_SPACING
            start_x = (display_width - total_width) / 2  # Start so that cards are centered
            if player == 0:
                start_y = 80
            else:
                start_y = 500
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

    def blackjack(self):

        self.dealer.calc_hand()
        self.player.calc_hand()

        show_dealer_card = pygame.image.load('img/' + self.dealer.card_img[1] + '.png').convert()
        
        if self.player.value == 21 and self.dealer.value == 21:
            # gameDisplay.blit(show_dealer_card, (550, 200))
            self.draw_cards(self.dealer.card_img, 0, True)
            black_jack("Both with BlackJack!", 500, 250, grey)
            time.sleep(4)
            self.play_or_exit()
        elif self.player.value == 21:
            # gameDisplay.blit(show_dealer_card, (550, 200))
            self.draw_cards(self.dealer.card_img, 0, True)
            black_jack("You got BlackJack!", 500, 250, green)
            time.sleep(4)
            self.play_or_exit()
        elif self.dealer.value == 21:
            # gameDisplay.blit(show_dealer_card, (550, 200))
            self.draw_cards(self.dealer.card_img, 0, True)
            black_jack("Dealer has BlackJack!", 500, 250, red)
            time.sleep(4)
            self.play_or_exit()
            
        self.player.value = 0
        self.dealer.value = 0
    
    # Load policy from input .policy file into self.policy
    def load_policy(self):
    # Policy not needed if a user is playing or a random policy is being used
        if self.action_type in ['random_policy', 'input']:
            return None
    # Read policy file and extract policy
        f = open(self.fixed_policy_filepath, 'r')
        data = f.read()
        data = data.split()
        policy = [int(x) for x in data]
        return policy
    
    def hand_to_state(self):
        self.dealer.calc_hand()
        self.player.calc_hand()

        # if self.state_mapping == 1:
        #  return self.sum_hand(player_hand) - 1

        if self.state_mapping == 2:
            return (self.player.value- 1) + (18 * (self.dealer.get_dealer_card() - 1))
        elif self.state_mapping == 3:
         if self.player.usable_ace() and len(self.player.cards) <= 2:
            return 181 + (self.player.value - 11) + (9 * (self.dealer.get_dealer_card() - 1))
         else:
            return (self.player.value - 1) + (18 * (self.dealer.get_dealer_card() - 1))
        
        self.player.value = 0
        self.dealer.value = 0
        
    
    def policy_run(self):
        if not self.dealer.cards:
           return

        state = self.hand_to_state()
        action = self.policy[state]

        time.sleep(2)
        if action: # hit: add a card to players hand and return
            game_texts('AI Hitting', display_width/2,650, True)
            self.hit()
        else: # stick: play out the dealers hand, and score
            game_texts('AI standing', display_width/2,650, True)

            self.stand()


    def deal(self):
        for i in range(2):
            self.dealer.add_card(self.deck.deal())
            self.player.add_card(self.deck.deal())

        # print(self.dealer.cards)
              
              
    
        self.dealer.display_cards()
        self.player.display_cards()
        self.player_card = 1
        # dealer_card = pygame.image.load('img/' + self.dealer.card_img[0] + '.png').convert()
        # dealer_card_2 = pygame.image.load('img/back.png').convert()
            
        # player_card = pygame.image.load('img/' + self.player.card_img[0] + '.png').convert()
        # player_card_2 = pygame.image.load('img/' + self.player.card_img[1] + '.png').convert()

        game_texts("Dealer's hand is:", display_width/2, 55)
        self.draw_cards(self.dealer.card_img, 0)

        # gameDisplay.blit(dealer_card, (400, 200))
        # gameDisplay.blit(dealer_card_2, (550, 200))

        game_texts("Your's hand is:", display_width/2, 475)
        
        self.draw_cards(self.player.card_img, 1)
        # gameDisplay.blit(player_card, (300, 450))
        # gameDisplay.blit(player_card_2, (410, 450))
        self.blackjack()
        
        
            
    
    def hit(self):
        self.player.add_card(self.deck.deal())
        self.blackjack()
        self.player_card += 1
                
        if self.player.value > 21:
            # show_dealer_card = pygame.image.load('img/' + self.dealer.card_img[1] + '.png').convert()
            self.draw_cards(self.dealer.card_img, 0, True)
            # gameDisplay.blit(show_dealer_card, (550, 200))
            game_finish("You Busted!", display_width/2, 250, red)
            time.sleep(4)
            self.play_or_exit()
        else:
            self.player.calc_hand()
            self.player.display_cards()
            # player_card_3 = pygame.image.load('img/' + self.player.card_img[2] + '.png').convert()
            # gameDisplay.blit(player_card_3, (520, 450))
            self.draw_cards(self.player.card_img, 1)
            
        self.player.value = 0
            
            
    def stand(self):
        # print("standing")
        # print(self.dealer.card_img)
        # print()
        # show_dealer_card = pygame.image.load('img/' + self.dealer.card_img[1] + '.png').convert()
        # gameDisplay.blit(show_dealer_card, (550, 200))
        self.draw_cards(self.dealer.card_img, 0, True)
        self.blackjack()
        self.dealer.calc_hand()
        self.player.calc_hand()
        if self.player.value > self.dealer.value:
            game_finish("You Won!", display_width/2, display_height/2, green)
            time.sleep(4)
            self.play_or_exit()
        elif self.player.value < self.dealer.value:
            game_finish("Dealer Wins!", display_width/2, display_height/2, red)
            time.sleep(4)
            self.play_or_exit()
        else:
            game_finish("It's a Tie!", display_width/2, display_height/2, grey)
            time.sleep(4)
            self.play_or_exit()
        
    
    def exit(self):
        sys.exit()
    
    def play_or_exit(self):
        game_texts("Play again press Deal!", display_width/2, 650)
        time.sleep(3)
        self.player.value = 0
        self.dealer.value = 0
        self.deck = Deck()
        self.dealer = Hand()
        self.player = Hand()
        self.deck.shuffle()
        gameDisplay.fill(background_color)
        # pygame.draw.rect(gameDisplay, grey, pygame.Rect(0, 0, 250, 700))
        pygame.display.update()
    
    

params = Params()     
play_blackjack = Play(params)

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        button("Deal", 770, 600, 120, 40, light_slat, dark_slat, play_blackjack.deal)
        # button("Hit", 30, 200, 150, 50, light_slat, dark_slat, play_blackjack.hit)
        # button("Stand", 30, 300, 150, 50, light_slat, dark_slat, play_blackjack.stand)
        button("EXIT", 770, 650, 120, 40, light_slat, dark_red, play_blackjack.exit)

        if params.action_type == 'fixed_policy':
           
            play_blackjack.policy_run()
            # time.sleep(5)
    
    pygame.display.flip()
