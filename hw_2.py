import numpy as np

alpha = 0.1
lamb = 0.5

n_states = 3
n_actions = 3
time_steps = 7

trajectory = [
        (1,2, 60/alpha),
        (0,0, 11/alpha - 60*lamb),
        (1,0,100),
        (0,1,60),
        (1,2,70),
        (2,1,40),
        (0,0,20),
        (2,2, np.nan)
        ]

Q = np.zeros((n_states, n_actions))
for i in range(len(trajectory)-1):
    s, a, r = trajectory[i]
    s_next = trajectory[i+1][0]
    Q[s,a] = Q[s,a] +alpha*(r + lamb*np.max(Q[s_next,:]) - Q[s,a])

print(f'Q-Learning: Q_7 =\n {Q}')
print(f'Q-Leaning, Best Policy; PI(A)={np.argmax(Q[0,:])},PI(B)={np.argmax(Q[1,:])},PI(C)={np.argmax(Q[2,:])}')

Q = np.zeros((n_states, n_actions))
for i in range(len(trajectory)-1):
    s, a, r = trajectory[i]
    s_next = trajectory[i+1][0]
    a_next = trajectory[i+1][1]
    Q[s,a]= Q[s,a] +alpha*(r + lamb*Q[s_next,] - Q[s,a])

print(f'Sarsa: Q_7 =\n {Q}')
print(f'Sarsa, Best Policy; PI(A)={np.argmax(Q[0,:])},PI(B)={np.argmax(Q[1,:])},PI(C)={np.argmax(Q[2,:])}')

