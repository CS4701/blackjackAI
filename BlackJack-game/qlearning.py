import os
import pandas as pd
import numpy as np
import time

GAMMA = 0.5
ALPHA = 0.01
LAMBDA = 0.1
STATES = 183
ACTION = 2

input= ''
output = ''

def update(Q_sa, df_i):
    diff = df_i.r + (GAMMA * max(Q_sa[df_i.sp])) - Q_sa[df_i.s][df_i.a]
    Q_sa[df_i.s][df_i.a] += ALPHA * diff
    return

def train(input):
    Q_sa = np.zeros((STATES, ACTION))
    df = pd.read_csv(input)
    for i in range(len(df)):
        df_i = df.loc[i]
        update(Q_sa, df_i)
    policy = np.argmax(Q_sa, axis=1)
    write_outfile(policy)
    return

def write_outfile(policy):
    '''
    Write policy to a .policy output file
    '''
    # Get output file name and path
    output_dir = os.getcwd()
    output_file = os.path.join(output_dir, f'{output}')

    # Open output file
    df = open(output_file, 'w')
    
    # Iterate through each value in policy, writing to output file
    for i in range(STATES):
        df.write(f'{policy[i]}\n')
        
    # Close output file
    df.close()
    
    return

def main():
    start = time.time()
    
    input_file = os.path.join(os.getcwd(), input)

    train(input_file)
    #train_q_lambda(input_file, CONST)

    end = time.time()

    print(f'Total time: {end-start:0.2f} seconds')
    print(f'Total time: {(end-start)/60:0.2f} minutes')
    
    return
if __name__ == '__main__':
    main()