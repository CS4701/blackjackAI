import random
from constants import *

class Deck:
    def __init__(self):
        self.cards = []
        self.build()
        self.shuffle()

    def build(self):
        for value in RANKS:
            for suit in SUITS:
                self.cards.append((value, suit))
  
    def shuffle(self):
        random.shuffle(self.cards)
        

    def deal(self):
        if len(self.cards) > 1:
            return self.cards.pop()
            
class Hand(Deck):
    def __init__(self, name="User", NN = False, wins = 0, params = None, policy = None):
        self.cards = []
        self.card_img = []
        self.value = 0 
        self.name = name
        self.busted = False
        self.wins = wins
        self.NN = NN
        if params:
            self.action_type = params.action_type
            self.fixed_policy_filepath = params.fixed_policy_filepath
            self.policy = policy
            self.state_mapping = params.state_mapping

    def add_card(self, card):
        self.cards.append(card)
    
    def useable_ace(self):
        total = 0
        first_card_index = [a_card[0] for a_card in self.cards]
        non_aces = [c for c in first_card_index if c != 'A']
        aces = [c for c in first_card_index if c == 'A']

        for card in non_aces:
            if card in 'JQK':
                total += 10
            else:
                total += int(card)
        return len(aces) >= 1 and total + 10 <= 21

    def calc_hand(self):
        first_card_index = [a_card[0] for a_card in self.cards]
        non_aces = [c for c in first_card_index if c != 'A']
        aces = [c for c in first_card_index if c == 'A']

        for card in non_aces:
            if card in 'JQK':
                self.value += 10
            else:
                self.value += int(card)

        for card in aces:
            if self.value <= 10:
                self.value += 11
            else:
                self.value += 1
    
    def get_dealer_card(self):
        # first_card_index = [a_card[0] for a_card in self.cards]
        # non_aces = [c for c in first_card_index if c != 'A']
        # aces = [c for c in first_card_index if c == 'A']
        # value = 0
        # for card in non_aces:
        #     if card in 'JQK':
        #         value += 10
        #     else:
        #         value += int(card)

        # for card in aces:
        #     if value <= 10:
        #         value += 11
        #     else:
        #         value += 1
        # return value

        first_card = self.cards[0]
        value = first_card[0]
        if value in 'JQK':
            return 10
        elif value == 'A':
            return 1
        else:
            return int(value)


    def display_cards(self):
        for card in self.cards:
            cards = "".join((card[0], card[1]))
            if cards not in self.card_img:
                self.card_img.append(cards)
    
    def reset(self):
        self.cards = []
        self.card_img = []
        self.value = 0 
        self.busted = False
