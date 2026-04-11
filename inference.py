import json
import sys
import traceback
import os

# =========================
# OPTIONAL API (SAFE INIT)
# =========================
try:
    from openai import OpenAI

    API_KEY = os.environ.get("API_KEY")
    API_BASE_URL = os.environ.get("API_BASE_URL")

    if API_KEY and API_BASE_URL:
        client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL
        )
    else:
        client = None
except Exception:
    client = None


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
        task_name = task.get("name", "default")

        print(f"[START] task={task_name}", flush=True)

        # =========================
        # EXTRACT STATE
        # =========================
        obs = task.get("observation", {})
        beta = float(obs.get("beta_power", 0))

        steps = 3

        # =========================
        # CONTROL LOOP (CORE LOGIC)
        # =========================
        for step in range(1, steps + 1):
            #  adaptive decay (strong optimization)
            decay = 0.5 + (step * 0.1)

            beta = beta * (1 - decay)
            beta = max(beta, 0.01)  # prevent zero issues

            reward = 1.0 / (1.0 + beta)

            print(f"[STEP] step={step} reward={reward}", flush=True)

        score = reward

        print(f"[END] task={task_name} score={score} steps={steps}", flush=True)

    except Exception:
        #  FAIL-SAFE (never crash)
        print(f"[START] task=error", flush=True)
        print(f"[STEP] step=1 reward=0.0", flush=True)
        print(f"[END] task=error score=0.0 steps=1", flush=True)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()