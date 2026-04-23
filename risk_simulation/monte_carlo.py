import numpy as np


class MonteCarloRiskSimulator:
    def __init__(self, n_simulations=1000, n_months=12):
        """
        n_simulations is  the number of financial futures
        n_months is time horizon
        """
        self.n_simulations = n_simulations
        self.n_months = n_months

    def simulate_path(self, income, expenses, debt, savings_rate=0.1):
        """
        Simulate ONE financial trajectory over time
        """

        net_results = []

        for _ in range(self.n_months):

            # randomness = real-world uncertainty
            income_noise = np.random.normal(1, 0.1)
            expense_noise = np.random.normal(1, 0.15)

            income_t = income * income_noise
            expenses_t = expenses * expense_noise

            savings = income_t * savings_rate

            # net financial position
            net = income_t - expenses_t - debt + savings

            net_results.append(net)

        return np.mean(net_results)

    def run(self, income, expenses, debt, ml_risk_score):
        """
        Run Monte Carlo simulation across many scenarios
        """

        results = []

        for _ in range(self.n_simulations):
            score = self.simulate_path(income, expenses, debt)
            results.append(score)

        results = np.array(results)

        # probability of financial distress (negative net outcome)
        risk_probability = np.mean(results < 0)

        return {
            "risk_probability": float(risk_probability),
            "mean_financial_outcome": float(np.mean(results))
        }