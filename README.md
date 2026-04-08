---
title: Neural DBS Environment Server
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - healthcare
---

# Neural DBS Environment (OpenEnv Hackathon)

# Overview

This project builds a "real-world reinforcement learning environment" inspired by Deep Brain Stimulation (DBS) used in Parkinson’s disease treatment.

Instead of fixed stimulation, this environment enables an AI agent to **adapt stimulation dynamically** based on neural activity.

---

# Problem Statement
  
Parkinson’s disease is linked to abnormal neural signals called "beta oscillations".
 
Current DBS systems:
- Use constant stimulation
- Do not adapt to patient condition
- Waste energy and can cause side effects

# Goal:
Train an agent to:
- Reduce pathological activity (beta power)
- Use minimal energy
- Maintain safe stimulation

---

# Environment Design

# Observation Space
- `beta_power` → neural signal severity (0–1)
- `phase` → oscillation phase
- `energy_used` → cumulative energy
- `time_step` → episode progress

---

   Action Space
- `amplitude` (0–1)
- `frequency` (0–1)
- `pulse_width` (0–1)

---

# Dynamics

beta_next = beta_current + drift - stimulation_effect + noise

- Drift → disease progression  
- Stimulation → reduces beta  
- Noise → real-world variability  

---

# Reward Function

reward = beta_reduction - energy_penalty - safety_penalty

Encourages:
- Lower beta power  
- Lower energy usage  
- Safe stimulation  

---

 # Tasks

| Task | Description |
|------|------------|
| Easy | Stable environment |
| Medium | Moderate noise |
| Hard | Highly dynamic system |

---

# Grading

Score range: **0.0 → 1.0**

- High score = effective + efficient control  
- Penalizes excessive energy  

---

# Key Highlights

- Real-world healthcare-inspired problem  
- Continuous control RL setup  
- Non-stationary environment  
- Safety-aware reward design  

---

# How to Run

# Test Environment
```bash
python test_env.py
Run Inference
python inference.py

Expected output:

[START]
[STEP]
[END]

# Docker
docker build -t neural-dbs-env .
docker run -p 8000:8000 neural-dbs-env

# API
POST /reset
POST /step
GET /state

# Baseline Performance

Using a simple fixed policy:

- Easy Task Score: ~0.93  
- Medium Task Score: ~0.93  
- Hard Task Score: ~0.93  

These scores demonstrate that the baseline policy can effectively reduce pathological activity, though more advanced RL methods can further optimize performance.

📁 Project Structure

neural_dbs_env/
├── models.py
├── tasks.py
├── graders.py
├── inference.py
├── test_env.py
├── openenv.yaml
├── server/
│   ├── app.py
│   ├── neural_dbs_env_environment.py
│   └── Dockerfile

Author
Shiva Malakummari

# redeploy trigger
