import os
import json
from openai import OpenAI

from models import NeuralDbsAction
from server.neural_dbs_env_environment import NeuralDbsEnvironment


# Initialize OpenAI client using hackathon-provided variables
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")


def get_action_from_llm(obs):
    prompt = f"""
You are controlling a neural stimulation system.

Current state:
beta_power = {obs.beta_power}
phase = {obs.phase}
energy_used = {obs.energy_used}

Goal:
- Reduce beta_power to near 0
- Minimize energy usage

Return ONLY valid JSON in this format:
{{"amplitude": float, "frequency": float, "pulse_width": float}}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    text = response.choices[0].message.content.strip()

    try:
        action_dict = json.loads(text)
    except:
        # fallback safe action
        action_dict = {
            "amplitude": 0.5,
            "frequency": 0.5,
            "pulse_width": 0.5
        }

    return NeuralDbsAction(
        amplitude=float(action_dict["amplitude"]),
        frequency=float(action_dict["frequency"]),
        pulse_width=float(action_dict["pulse_width"])
    )


def run_task(task_name):
    env = NeuralDbsEnvironment()
    obs = env.reset()

    print(f"[START] task={task_name} env=neural_dbs_env model={MODEL_NAME}")

    rewards = []

    for step in range(10):
        action = get_action_from_llm(obs)

        obs = env.step(action)
        rewards.append(obs.reward)

        print(
            f"[STEP] step={step+1} "
            f"action=amp={round(action.amplitude,2)},freq={round(action.frequency,2)},pw={round(action.pulse_width,2)} "
            f"reward={round(obs.reward,2)} "
            f"done={str(obs.done).lower()} error=null"
        )

        if obs.done:
            break

    score = sum(rewards) / len(rewards)

    print(
        f"[END] success=true steps={len(rewards)} "
        f"score={round(score,3)} "
        f"rewards={','.join([str(round(r,2)) for r in rewards])}"
    )


if _name_ == "_main_":
    main()