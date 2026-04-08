from server.neural_dbs_env_environment import NeuralDbsEnvironment
from models import NeuralDbsAction

env = NeuralDbsEnvironment()
obs = env.reset()
print(obs)

for _ in range(5):
    action = NeuralDbsAction(amplitude=0.5, frequency=0.5, pulse_width=0.5)
    obs = env.step(action)
    print(obs)

from graders import grade

score = grade(env)
print("Final Score:", score)