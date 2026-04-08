def grade(env):
    """
    Compute a score between 0.0 and 1.0 based on:

    - Lower beta_power (better symptom control)
    - Lower energy_used (efficiency)
    """

    # Normalize beta score (lower beta is better)
    beta_score = 1.0 - env.beta_power  # already between 0–1

    # Normalize energy (assume max reasonable energy ~10)
    energy_penalty = min(1.0, env.energy_used / 10.0)

    # Final score
    score = beta_score - 0.5 * energy_penalty

    # Clamp between 0 and 1
    score = max(0.0, min(1.0, score))

    return score