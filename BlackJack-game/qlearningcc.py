import os
import pandas as pd
import numpy as np
import time

GAMMA = 0.5  # Discount factor
ALPHA = 0.01  # Learning rate
STATES = 183 * 21 * 21  # (initial_hand, dealer_up, run_count)
ACTIONS = 2  # (Hit, Stand)
LAMBDA = 0.1


input= '/Users/darrenchoy/Desktop/GitHub/blackjackAI/blackjack_simulator.csv'
output = '/Users/darrenchoy/Desktop/GitHub/blackjackAI/qlearningcc.policy'

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
    for i in range(int(len(df))):
        df_i = df.loc[i]
        clean_str = df_i.initial_hand.strip("[]").replace(",", "")
        initial_hand = [int(part) for part in clean_str.split() if part.strip()]
        dealer_up = df_i.dealer_up
        run_count = df_i.run_count
        actions_taken = df_i.actions_taken.strip("[]")
        win = df_i.win

        state = encode_state(initial_hand, dealer_up, run_count)
        print("State: ", state)
        if actions_taken != "":
            final_action = actions_taken[1] 
        else:
            final_action == 'S'
        if final_action == 'H' or final_action == 'P' or final_action == 'D':
            action = 1
        elif final_action == 'S' or final_action == 'R':
            action = 0
        else:
            continue  # Skip if action is not recognized
        reward = win  # Assign reward for the action
        
        # Update Q-value
        print("updating Q-value")
        Q_sa[state, action] += ALPHA * (reward + GAMMA * np.max(Q_sa[state]) - Q_sa[state, action])

    policy = np.argmax(Q_sa, axis=1)

    # Write policy to file
    write_outfile(policy)


    return

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