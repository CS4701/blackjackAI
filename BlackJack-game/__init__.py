import pygame as pygame
import blackjack_deck
import constants 
import sys
import time
pygame.init()

clock = pygame.time.Clock()

gameDisplay = pygame.display.set_mode((constants.display_width, constants.display_height))

pygame.display.set_caption('BlackJack')
gameDisplay.fill(constants.background_color)
pygame.draw.rect(gameDisplay, constants.grey, pygame.Rect(0, 0, 250, 700))

###text object render
def text_objects(text, font):
    textSurface = font.render(text, True, constants.black)
    return textSurface, textSurface.get_rect()

def end_text_objects(text, font, color):
    textSurface = font.render(text, True, color)
    return textSurface, textSurface.get_rect()

play_blackjack = Play()

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        blackjack_deck.button("Deal", 30, 100, 150, 50, constants.light_slat, constants.dark_slat, play_blackjack.deal)
        blackjack_deck.button("Hit", 30, 200, 150, 50, constants.light_slat, constants.dark_slat, play_blackjack.hit)
        blackjack_deck.button("Stand", 30, 300, 150, 50, constants.light_slat, constants.dark_slat, play_blackjack.stand)
        blackjack_deck.button("EXIT", 30, 500, 150, 50, constants.light_slat, constants.dark_red, play_blackjack.exit)
    
    pygame.display.flip()