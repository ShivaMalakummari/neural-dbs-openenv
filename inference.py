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
State:
beta_power={obs.beta_power}
phase={obs.phase}
energy={obs.energy_used}

Goal:
Reduce beta_power to 0 with minimal energy.

Return ONLY JSON:
{{"amplitude": float, "frequency": float, "pulse_width": float}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content.strip()

        action_dict = json.loads(text)

        return NeuralDbsAction(
            amplitude=float(action_dict["amplitude"]),
            frequency=float(action_dict["frequency"]),
            pulse_width=float(action_dict["pulse_width"])
        )

    except Exception as e:
        #  CRITICAL: NEVER CRASH
        return NeuralDbsAction(
            amplitude=0.5,
            frequency=0.5,
            pulse_width=0.5
        )


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)