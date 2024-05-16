import os
import pandas as pd
import numpy as np

def setup_environment(state_mapping):
    """ Sets up parameters based on the state mapping provided """
    global S, A, gam, alpha, maxIters, input_file, output_file
    if state_mapping == 1:
        S = 21
        A = 2
        gam = 0.5
        alpha = 0.01
        maxIters = 5
        input_file = "random_policy_runs_mapping_1.csv"
        output_file = 'Sarsa_Policy_1.policy'
    elif state_mapping == 2:
        S = 183
        A = 2
        gam = 0.5
        alpha = 0.01
        maxIters = 5
        input_file = "random_policy_runs_mapping_2.csv"
        output_file = 'Sarsa_Policy_2.policy'

def load_data():
    """ Load data from the input file and return it """
    df = pd.read_csv(input_file)
    return df['s'], df['a'], df['r'], df['sp'], df['a'].shift(-1)  # Include next action directly

def initialize_action_value_function():
    """ Initialize the action value function Q to zeros """
    return np.zeros((S, A))

def perform_sarsa(s_data, a_data, r_data, sp_data, ap_data, Q):
    """ Perform the SARSA updates on Q """
    for i in range(maxIters):
        print(f"Iteration {i+1}/{maxIters}")
        for k in range(len(s_data)-1):  # Exclude the last item due to shifted 'ap_data'
            s, a, r, sp, ap = int(s_data[k]), int(a_data[k]), r_data[k], int(sp_data[k]), int(ap_data[k])
            Q[s, a] += alpha * (r + gam * Q[sp, ap] - Q[s, a])

def extract_policy(Q):
    """ Extract the policy from Q """
    return np.argmax(Q, axis=1)

def write_policy_to_file(policy):
    """ Write the policy to a file """
    output_dir = os.getcwd()
    output_file_path = os.path.join(output_dir, output_file)
    with open(output_file_path, 'w') as file:
        for p in policy:
            file.write(f'{p}\n')

def main(state_mapping):
    setup_environment(state_mapping)
    s_data, a_data, r_data, sp_data, ap_data = load_data()
    Q = initialize_action_value_function()
    perform_sarsa(s_data, a_data, r_data, sp_data, ap_data, Q)
    policy = extract_policy(Q)
    write_policy_to_file(policy)

if __name__ == "__main__":
    state_mapping = 2  # Define the state mapping
    main(state_mapping)
