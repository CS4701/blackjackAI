import os
import pandas as pd
import numpy as np

# Define global variables
S, A, gam, maxIters, input_file, output_file = None, None, 0.5, 100, None, None

def setup_environment(state_mapping):
    global S, A, gam, maxIters, input_file, output_file
    if state_mapping == 1:
        S = 21
        input_file = "random_policy_runs_mapping_1.csv"
        output_file = 'Value_Iteration_Policy_1.policy'
    elif state_mapping == 2:
        S = 183
        input_file = "random_policy_runs_mapping_2.csv"
        output_file = 'Value_Iteration_Policy_2.policy'
    A = 2  # Action space size is constant in both cases

def initialize_matrices():
    U = np.zeros(S)
    T = np.zeros((S, S, A))
    R = np.zeros((S, A))
    N = np.zeros((S, A, S))
    policy = np.zeros(S, dtype=int)
    return U, T, R, N, policy

def update_model(df, U, T, R, N):
    for k in range(len(df)):
        s, a, r, sp = int(df.iloc[k]['s']), int(df.iloc[k]['a']), df.iloc[k]['r'], int(df.iloc[k]['sp'])
        N[s, a, sp] += 1
        if sum(N[s, a, :]) != 0:
            T[sp, s, a] = N[s, a, sp] / sum(N[s, a, :])
            R[s, a] = (R[s, a] * (N[s, a, sp] - 1) + r) / N[s, a, sp]

def perform_value_iteration(U, T, R, policy):
    for s in range(S):
        for _ in range(maxIters):
            u = np.zeros(A)  # Reinitialize u for each state and iteration
            for a in range(A):
                u[a] = R[s, a] + gam * np.sum(T[:, s, a] * U)
            U[s] = np.max(u)
            policy[s] = np.argmax(u)

def write_policy_to_file(policy, output_file):
    output_dir = os.getcwd()
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as file:
        for p in policy:
            file.write(f'{p}\n')

def main(state_mapping):
    setup_environment(state_mapping)
    df = pd.read_csv(input_file)
    U, T, R, N, policy = initialize_matrices()
    update_model(df, U, T, R, N)
    perform_value_iteration(U, T, R, policy)
    write_policy_to_file(policy, output_file)

if __name__ == "__main__":
    state_mapping = 1  # Set state mapping here
    main(state_mapping)
