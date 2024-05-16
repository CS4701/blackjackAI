import os
import pandas as pd
import numpy as np
import time

class QLearningConstants:
    """Class that contains various constants/parameters for the Q-learning problem."""
    def __init__(self, state_mapping):
        self.gamma = 0.5
        self.alpha = 0.01
        self.lambda_ = 0.1
        if state_mapping == 1:
            self.n_states = 21
            self.input_filename = 'random_policy_runs_mapping_1.csv'
            self.output_filename = 'QLearning_policy_mapping_1.policy'
        elif state_mapping == 2:
            self.n_states = 183
            self.input_filename = 'random_policy_runs_mapping_2.csv'
            self.output_filename = 'QLearning_policy_mapping_2.policy'
        self.n_action = 2

def update_q_learning(Q, s, a, r, sp, constants):
    """Perform Q-Learning update to action value function for a single transition."""
    max_next_q = np.max(Q[sp])
    Q[s, a] += constants.alpha * (r + constants.gamma * max_next_q - Q[s, a])

def train_q(input_file, constants):
    """Train a policy using the Q-learning algorithm and input datafile containing sample data."""
    df = pd.read_csv(input_file)
    Q = np.zeros((constants.n_states, constants.n_action))
    for index, row in df.iterrows():
        update_q_learning(Q, int(row['s']), int(row['a']), row['r'], int(row['sp']), constants)
    return np.argmax(Q, axis=1)

def write_outfile(policy, output_file):
    """Write policy to a .policy output file."""
    output_dir = os.getcwd()
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as file:
        for p in policy:
            file.write(f'{p}\n')

def main():
    start_time = time.time()

    # Set state mapping here (1 or 2)
    state_mapping = 1
    constants = QLearningConstants(state_mapping)
    input_file = os.path.join(os.getcwd(), constants.input_filename)
    policy = train_q(input_file, constants)
    write_outfile(policy, constants.output_filename)

    end_time = time.time()
    print(f'Total time: {end_time - start_time:.2f} seconds')

if __name__ == '__main__':
    main()
