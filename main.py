import pyro.distributions as dist
import pyro
import torch

PRIOR = {"mu_log2_skill": -0.29, "sigma_log2_skill": 0.1}


def model(data):
    mu_pior = torch.tensor(PRIOR["mu_log2_skill"])
    sigma_pior = torch.tensor(PRIOR["sigma_log2_skill"])

    latent_skill[data.reference_player] = 1.0  # TODO: maybe I have to put an array
    latent_skill = {
        p: pyro.sample(
            f"latent_skill_{p}", dist.LogNormal(loc=mu_pior, scale=sigma_pior)
        )
        for p in data.players(include_reference_player=False)
    }

    for m, i in enumerate(data.matches()):
        latent_skill_team_a = sum(latent_skill[p] for p in m.team_a())
        latent_skill_team_b = sum(latent_skill[p] for p in m.team_b())
        prob_A = latent_skill_team_a / (latent_skill_team_a + latent_skill_team_b)
        pyro.sample(
            f"match_{i+1}",
            dist.Binomial(total_count=m.total_goals(), probs=prob_A),
            obs=m.goals_A(),
        )

    def guide(data):
        # register the two variational parameters with Pyro.
        alpha_q = pyro.param(
            "alpha_q", torch.tensor(15.0), constraint=constraints.positive
        )
        beta_q = pyro.param(
            "beta_q", torch.tensor(15.0), constraint=constraints.positive
        )
        # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
        pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))
