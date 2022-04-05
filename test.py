from mdp import MDP

# TEST
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
c = [23, 24, 25]

mdp = MDP(states=[a, b, c], actions=[1,2,3,4])

for i in range(4*4*3):
  state = mdp.get_state(i)
  real_state = mdp.get_real_state_value(i)
  index = mdp.get_index(real_state)
  if i != index:
    raise ValueError("Something is wrong")
  if i%10==0:
    print(i, state, real_state, index)

print("Everything is correct!")