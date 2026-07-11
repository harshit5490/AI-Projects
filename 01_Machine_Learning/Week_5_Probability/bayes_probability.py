# ==========================================
# Bayes Probability Calculator
# Week 6 - Probability
# AI Engineer Bootcamp
# ==========================================

def bayes(prior, likelihood, evidence):
    """
    Calculate Posterior Probability using Bayes' Theorem

    Formula:
    P(A|B) = (P(B|A) * P(A)) / P(B)
    """

    posterior = (likelihood * prior) / evidence
    return posterior


# ==========================================
# Example 1 : Medical Diagnosis
# ==========================================

print("=" * 50)
print("Medical Diagnosis Example")
print("=" * 50)

prior = 0.02               # P(Disease)
likelihood = 0.95          # P(Positive | Disease)
evidence = 0.08            # P(Positive)

posterior = bayes(prior, likelihood, evidence)

print(f"Prior Probability      : {prior}")
print(f"Likelihood             : {likelihood}")
print(f"Evidence               : {evidence}")
print(f"Posterior Probability  : {posterior:.4f}")

print()

# ==========================================
# Example 2 : Spam Detection
# ==========================================

print("=" * 50)
print("Spam Detection Example")
print("=" * 50)

prior = 0.40               # P(Spam)
likelihood = 0.90          # P(FREE | Spam)
evidence = 0.50            # P(FREE)

posterior = bayes(prior, likelihood, evidence)

print(f"Prior Probability      : {prior}")
print(f"Likelihood             : {likelihood}")
print(f"Evidence               : {evidence}")
print(f"Posterior Probability  : {posterior:.4f}")

print()

# ==========================================
# User Input
# ==========================================

print("=" * 50)
print("Bayes Probability Calculator")
print("=" * 50)

prior = float(input("Enter Prior Probability: "))
likelihood = float(input("Enter Likelihood: "))
evidence = float(input("Enter Evidence: "))

posterior = bayes(prior, likelihood, evidence)

print("\nResult")
print("-" * 30)
print(f"Posterior Probability = {posterior:.4f}")