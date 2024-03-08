from blackjack_deck import *

deck = Deck()
deck.shuffle()

player = Hand()
dealer = Hand()

for i in range(2):
    player.add_card(deck.deal())
    dealer.add_card(deck.deal())

print(f"Player Hand: {player.cards}")
print(f"Dealer Hand: {dealer.cards}")


player.add_card(deck.deal())

print(f"Player Hand after adding card: {player.cards}")

player.calc_hand()
dealer.calc_hand()



