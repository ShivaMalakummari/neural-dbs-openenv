import json
import sys
import traceback
import os
from openai import OpenAI

# =========================
# SAFE CLIENT INITIALIZATION
# =========================
API_KEY = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL")

if API_KEY and API_BASE_URL:
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL
    )
else:
    client = None  # fallback for local testing

MODEL_NAME = "gpt-4o-mini"


# =========================
# ACTION CLASS
# =========================
class NeuralDbsAction:
    def __init__(self, amplitude, frequency, pulse_width):
        self.amplitude = amplitude
        self.frequency = frequency
        self.pulse_width = pulse_width


# =========================
# LLM FUNCTION (SAFE)
# =========================
def get_action_from_llm(obs):
    # If no API available → fallback
    if client is None:
        return NeuralDbsAction(0.5, 0.5, 0.5)

    prompt = f"""
State:
beta_power={obs.get('beta_power', 0)}
phase={obs.get('phase', 0)}
energy={obs.get('energy_used', 0)}

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
            float(action_dict["amplitude"]),
            float(action_dict["frequency"]),
            float(action_dict["pulse_width"])
        )

    except Exception:
        # fallback if API fails
        return NeuralDbsAction(0.5, 0.5, 0.5)


# =========================
# CORE TASK FUNCTION
# =========================
def run_task(task):
    try:
        obs = task.get("observation", task)

        action = get_action_from_llm(obs)

        return {
            "amplitude": action.amplitude,
            "frequency": action.frequency,
            "pulse_width": action.pulse_width
        }

    except Exception as e:
        return {
            "amplitude": 0.5,
            "frequency": 0.5,
            "pulse_width": 0.5,
            "error": str(e)
        }


# =========================
# MAIN FUNCTION
# =========================
def main():
    try:
        input_data = sys.stdin.read()

        if not input_data:
            raise ValueError("No input provided")

        data = json.loads(input_data)

        if "task" not in data:
            raise KeyError("Missing 'task' key")

        task = data["task"]

        output = run_task(task)

        print(json.dumps(output))

    except Exception as e:
        # NEVER CRASH
        print(json.dumps({
            "amplitude": 0.5,
            "frequency": 0.5,
            "pulse_width": 0.5,
            "error": str(e),
            "traceback": traceback.format_exc()
        }))


# =========================
# ENTRY POINT (CORRECT)
# =========================
if __name__ == "__main__":
    main()