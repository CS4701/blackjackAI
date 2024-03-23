import os
import pandas as pd
import numpy as np
import time

GAMMA = 0.5  # Discount factor
ALPHA = 0.01  # Learning rate
STATES = 183 * 21 * 21  # (initial_hand, dealer_up, run_count)
ACTIONS = 2  # (Hit, Stand)
LAMBDA = 0.1


input= ''
output = ''
def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

# Return current hand total 
def sum_hand(hand):
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)
def encode_state(initial_hand, dealer_up, run_count):
    player_hand_value = sum_hand(initial_hand)
    return player_hand_value + dealer_up * 21 + (run_count + 10) * 21 * 21

def train(input_file):
    Q_sa = np.zeros((STATES, ACTIONS))
    df = pd.read_csv(input_file)

    for _, row in df.iterrows():
        initial_hand = eval(row['initial_hand'])
        dealer_up = row['dealer_up']
        run_count = row['run_count']
        actions_taken = eval(row['actions_taken'])
        win = row['win']

        state = encode_state(initial_hand, dealer_up, run_count)

        for action in actions_taken:
            next_state = state  # Assuming the state doesn't change within a single hand
            reward = win if action == 'S' else 0  # Assign reward only for the final action

            # Update Q-value using the Q-learning update rule
            Q_sa[state, action] += ALPHA * (reward + GAMMA * np.max(Q_sa[next_state]) - Q_sa[state, action])

    return Q_sa

def write_outfile(policy):
    '''
    Write policy to a .policy output file
    '''
    output_dir = os.getcwd()
    output_file = os.path.join(output_dir, f'{output}')

    df = open(output_file, 'w')
    
    for i in range(STATES):
        df.write(f'{policy[i]}\n')
        
    # Close output file
    df.close()
    
    return

def main():
    start = time.time()
    
    input_file = os.path.join(os.getcwd(), input)

    train(input_file)

    end = time.time()

    print(f'Total time: {end-start:0.2f} seconds')
    print(f'Total time: {(end-start)/60:0.2f} minutes')
    
    return
if __name__ == '__main__':
    main()