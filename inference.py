import os
from openai import OpenAI

from server.neural_dbs_env_environment import NeuralDbsEnvironment
from models import NeuralDbsAction
from graders import grade
from tasks import TASKS

# Required env variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline-policy")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def run_task(env, task_name, task_config):
    rewards = []
    steps = 0

    print(f"[START] task={task_name} env=neural_dbs_env model={MODEL_NAME}")

    try:
        obs = env.reset(task_config)

        for step in range(1, 11):
            action = NeuralDbsAction(
                amplitude=0.5,
                frequency=0.5,
                pulse_width=0.5
            )

            result = env.step(action)

            reward = result.reward
            done = result.done
            error = None

            rewards.append(reward)
            steps = step

            print(
                f"[STEP] step={step} action=amp=0.5,freq=0.5,pw=0.5 "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )

            if done:
                break

        score = grade(env)
        success = score > 0.5

    except Exception as e:
        success = False
        score = 0.0

    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}"
    )


if __name__ == "__main__":
    env = NeuralDbsEnvironment()

    for task_name, task_config in TASKS.items():
        run_task(env, task_name, task_config)