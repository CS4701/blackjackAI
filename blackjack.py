import os
import numpy as np
import pandas as pd
import random

class Params():
  def __init__(self):
  # 'input', 'random_policy', 'fixed_policy'
    self.action_type = 'fixed_policy'
  # Only used for 'random_policy' or 'fixed_policy' input
    self.num_games = 20000
  # Filepath to fixed policy file (only used for 'fixed_policy' input)
    # self.fixed_policy_filepath = os.path.join(os.getcwd(), 'Value_Iteration_policy_1.policy')
    # self.fixed_policy_filepath = os.path.join(os.getcwd(), 'Sarsa_policy_2.policy')
    self.fixed_policy_filepath = os.path.join(os.getcwd(), 'QLearning_policy_mapping_1.policy')

    # self.fixed_policy_filepath = os.path.join(os.getcwd(), 'model_predictions2.policy') 
    # self.fixed_policy_filepath = os.path.join(os.getcwd(), 'math_policy.policy')

  # Which state mapping algorithm to use (1 or 2)
    self.state_mapping = 1
    return

"""
State Mapping 1: state = players_hand - 1
State 0 - lose state
State 1 - win state
State 2 - terminal state
State 3 - players hand sums to 4
...
State 19 - players hand sums to 20
State 20 - players hand sums to 21
-------------------------------------------------------------------------------
State Mapping 2: state = (players_hand - 1) + (18 * (dealers_hand-1))
State 0 - lose state
State 1 - win state
State 2 - terminal state
State 3 - players hand sums to 4, dealers hand is 1
State 4 - players hand sums to 5, dealers hand is 1
...
State 19 - players hand sums to 20, dealers hand is 1
State 20 - players hand sums to 21, dealers hand is 1
State 21 - players hand sums to 4, dealers hand is 2
State 22 - players hand sums to 5, dealers hand is 2
...
State 181 - players hand sums to 20, dealers hand is 10
State 182 - players hand sums to 21, dealers hand is 10

-------------------------------------------------------------------------------
State Mapping 3: (need to add new states for card counting) state should now be about 54 * 180

State 0 - lose state
State 1 - win state
State 2 - terminal state
State 3 - players hand sums to 4, dealers hand is 1, 
State 4 - players hand sums to 5, dealers hand is 1, 
...
State 19 - players hand sums to 20, dealers hand is 1
State 20 - players hand sums to 21, dealers hand is 1
State 21 - players hand sums to 4, dealers hand is 2
State 22 - players hand sums to 5, dealers hand is 2
...
State 181 - players hand sums to 20, dealers hand is 10
State 182 - players hand sums to 21, dealers hand is 10
...
State 183 - players has Ace and 2, dealers hand is 1,
State 184 - players has Ace and 3, dealers hand is 1,
...
State 190 - players has Ace and 9, dealers hand is 1
State 191 - players has Ace and 10, dealers hand is 1
State 192 - players has Ace and 2, dealers hand is 2
State 193 - players has Ace and 3, dealers hand is 2
...
State 271 - player has Ace and 9, dealers hand is 10
State 272 - players has Ace and 10, dealers hand is 10
...

"""

class BlackJack_game():
  def __init__(self, params):
    # 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
    self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]*4
    random.shuffle(self.deck)
    # Player and dealer hands
    self.player = self.draw_hand()
    self.dealer = [self.draw_card()]

    self.sarsp = []
    self.sarsp_arr = np.array([], dtype='int').reshape(0,4)

    self.action_type = params.action_type # 'input', 'random_policy', 'fixed_policy'
    self.verbose = (params.action_type == 'input')
    self.num_games = params.num_games
    self.fixed_policy_filepath = params.fixed_policy_filepath
    self.policy = self.load_policy()
    self.state_mapping = params.state_mapping

    self.lose_state = 0
    self.win_state = 1
    self.terminal_state = 2
    self.lose_reward = -10
    self.win_reward = 10
    self.run_count = 0
    return


# Reset deck, player/dealer hands, and sarsp for a new game
  def reset(self):
    self.player = self.draw_hand()
    self.dealer = [self.draw_card()]
    self.sarsp = []
    self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]*4
    random.shuffle(self.deck)
    return

  def draw_card(self):
    return self.deck.pop()


  def draw_hand(self):
    return [self.draw_card(), self.draw_card()]
    
  def usable_ace(self, hand):
    return 1 in hand and sum(hand) + 10 <= 21

  def sum_hand(self, hand):
    if self.usable_ace(hand):
      return sum(hand) + 10
    return sum(hand)

  def is_bust(self, hand):
    return self.sum_hand(hand) > 21

  def score(self, hand):
    return 0 if self.is_bust(hand) else self.sum_hand(hand)

  def player_won(self, player, dealer):
    if self.is_bust(player):
      return False
    elif self.is_bust(dealer):
      return True
    elif self.sum_hand(player) > self.sum_hand(dealer):
      return True
    else:
      return False
    
  def encode_state(self, player_hand, dealer_up, run_count):
    for card in player_hand:
        if card >= 10 and card <= 1:
            run_count -= 1
        elif card in [7, 8, 9]:
            pass
        else:
            run_count += 1

    if dealer_up >= 10 and dealer_up <= -1:
        run_count += 1
    elif dealer_up in [7, 8, 9]:
        pass
    else:
        run_count -= 1

    return self.sum_hand(player_hand) + dealer_up * 21 + (run_count + 10) * 21 * 21

  def hand_to_state(self, player_hand, dealer):
    if self.state_mapping == 1:
      return self.sum_hand(player_hand) - 1
    elif self.state_mapping == 2:
      return (self.sum_hand(player_hand) - 1) + (18 * (dealer[0] - 1))
    elif self.state_mapping == 3:
      if self.usable_ace(player_hand) and len(player_hand) <= 2:
        return 181 + (self.sum_hand(player_hand) - 11) + (9 * (dealer[0] - 1))
      else:
        return (self.sum_hand(player_hand) - 1) + (18 * (dealer[0] - 1))
    elif self.state_mapping == 4:
      return self.encode_state(player_hand, dealer[0], self.run_count)
        
  def get_reward(self, state, action, player, dealer):
    if self.state_mapping == 1:
      return 0
    else:
      if ((self.sum_hand(player) <= 11 and action == 1) or (self.sum_hand(player) >= 17 and action == 0)):
        return 1
      elif ((self.sum_hand(player) <= 11 and action == 0) or (self.sum_hand(player) >= 17 and action == 1)):
        return -1
      else:
        return 0
      
  def load_policy(self):
    if self.action_type in ['random_policy', 'input']:
      return None
    f = open(self.fixed_policy_filepath, 'r')
    data = f.read()
    data = data.split()
    policy = [int(x) for x in data]
    return policy
    
  def print_iter(self):
    if not self.verbose:
      return
    print(f'Player hand: {self.player}\t\t sum: {self.sum_hand(self.player)}')
    print(f'Dealer hand: {self.dealer}\t\t sum: {self.sum_hand(self.dealer)}')
    return

  def get_action(self, state):
    if self.action_type == 'input':
      action = int(input('Hit (1) or Pass (0): '))
    elif self.action_type == 'random_policy':
      action = np.random.randint(2)
    elif self.action_type == 'fixed_policy':
      action = self.policy[state]
    return action

  def play_game(self):
    if self.verbose:
      print('New Game!\n')
    done = False
    while(not done):
      self.print_iter()
      state = self.hand_to_state(self.player, self.dealer)
      action = self.get_action(state)
      reward = self.get_reward(state, action, self.player, self.dealer)
      if action: # hit: add a card to players hand and return
        self.player.append(self.draw_card())
        if self.is_bust(self.player):
          done = True
        else:
          done = False
      else: # stick: play out the dealers hand, and score
        while self.sum_hand(self.dealer) < 17:
          self.dealer.append(self.draw_card())
          done = True
    
    if(not done):
      sp = self.hand_to_state(self.player, self.dealer)
      self.sarsp.append([state, action, reward, sp])

    self.print_iter()
    player_won_bool = self.player_won(self.player, self.dealer)
    if player_won_bool:
      sp = self.win_state
    else:
      sp = self.lose_state
    self.sarsp.append([state, action, reward, sp])
    
    state = sp

    if player_won_bool:
      reward = self.win_reward
    else:
      reward = self.lose_reward
    self.sarsp.append([state, np.random.randint(2), reward, self.terminal_state])
    
    if self.verbose:
      print(f'Player won?: {player_won_bool}')
          
    self.sarsp_arr = np.vstack((self.sarsp_arr, np.array(self.sarsp)))

    return

  def output_sarsp_file(self):
    filename = f'random_policy_runs_mapping_{self.state_mapping}.csv'

    output_filepath = os.path.join(os.getcwd(), filename)
    header = ['s', 'a', 'r', 'sp']
    pd.DataFrame(self.sarsp_arr).to_csv(output_filepath, header=header, index=None)
    return

  def print_stats(self):
    num_wins = np.count_nonzero(self.sarsp_arr[:,0] == self.win_state)
    num_lose = np.count_nonzero(self.sarsp_arr[:,0] == self.lose_state)

    print(f'Number of games: {self.num_games}')
    print(f'Number of wins: {num_wins}')
    print(f'Number of losses: {num_lose}')
    print(f'Win Percentage: {num_wins / self.num_games : .3f}')
    
    return
    
  # Simulate (num_games) games of BlackJack!
  def play_games(self):
    #Iterate through num_games
    for i in range(self.num_games):
      self.play_game()
      self.reset()

    # print(self.sarsp_arr)
    self.print_stats()
    
    if self.action_type == 'random_policy':
      self.output_sarsp_file()
      
    return



def main():

  # Input parameters
  params = Params()
  assert (params.action_type in ['input', 'fixed_policy', 'random_policy']), "Action type must be 'input', 'fixed_policy', or 'random_policy'"

  game = BlackJack_game(params)
  # policy is being used
  if params.action_type == 'input':
    game.play_game()
  else:
    game.play_games()
    
  return
  
if __name__ == "__main__": 
  main()
                              


