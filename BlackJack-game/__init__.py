# import pygame as pygame
# from blackjack_deck import *
# from constants import *
# import sys
# import time
# pygame.init()

# clock = pygame.time.Clock()

# gameDisplay = pygame.display.set_mode((display_width, display_height))

# pygame.display.set_caption('BlackJack')
# gameDisplay.fill(background_color)
# pygame.draw.rect(gameDisplay, grey, pygame.Rect(0, 0, 250, 700))

# ###text object render
# def text_objects(text, font):
#     textSurface = font.render(text, True, black)
#     return textSurface, textSurface.get_rect()

# def end_text_objects(text, font, color):
#     textSurface = font.render(text, True, color)
#     return textSurface, textSurface.get_rect()

# play_blackjack = Play()

# running = True

# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#         button("Deal", 30, 100, 150, 50, light_slat, dark_slat, play_blackjack.deal)
#         button("Hit", 30, 200, 150, 50, light_slat, dark_slat, play_blackjack.hit)
#         button("Stand", 30, 300, 150, 50, light_slat, dark_slat, play_blackjack.stand)
#         button("EXIT", 30, 500, 150, 50, light_slat, dark_red, play_blackjack.exit)
    
#     pygame.display.flip()