"""
Verifying feasible wards of the actual election data against
random uniform data based on the log likelihood distribution of
the n least likely candidates.
"""
import numpy as np
import pandas as pd
from HU.preprocessing import get_preprocessed_data
from HU.app5_ent_in_top import plot_entropy_distribution
from arguments import is_quiet, save_output, load_output
from drdigit.digit_entropy_distribution import LodigeTest
from drdigit.digit_filtering import get_feasible_groups
import HU.robustness


# TODO: remove via refactoring into dependency
_DEFAULT_ITERATIONS = 12345
_DEFAULT_RANDOM_SEED = 1234
_DEFAULT_PE_RANDOM_SEED = 1234
_DEFAULT_PE_ITERATIONS = 10000


def save_results(actual_likelihood, probabilities, sample):
    df = pd.DataFrame(dict(names=["actual_likelihood"],
                           values=[actual_likelihood]))
    save_output(df, "app9_likelihood.csv")

    df = pd.DataFrame(dict(probability=probabilities))
    save_output(df, "app9_probabilities.csv")

    df = pd.DataFrame(dict(likelihood=sample))
    save_output(df, "app9_sample.csv")


def load_results():
    actual_likelihood = \
        load_output("app9_likelihood.csv").iloc[0]["values"]
    probabilities = \
        load_output("app9_probabilities.csv").probability
    sample = load_output("app9_sample.csv").likelihood

    return actual_likelihood, probabilities, sample


def run_simulation(bottom_n,
                   seed=None,
                   seeds=None,
                   iterations=_DEFAULT_ITERATIONS,
                   pe_seed=_DEFAULT_PE_RANDOM_SEED,
                   pe_iterations=_DEFAULT_PE_ITERATIONS,
                   party_name="Fidesz"):

    if seeds is None:
        if seed is None:
            seeds = [_DEFAULT_RANDOM_SEED]
        else:
            seeds = [seed]
    else:
        assert seed is None, "Only seed or seeds should be specified not both"
    del seed


    df = get_preprocessed_data()
    feasible_settlements = \
        get_feasible_groups(df, min_n_wards=8, min_votes=100)
    print("Found", len(feasible_settlements), "feasible settlements")
    df = df[df["Telepules"].isin(feasible_settlements)]
    df = df.sort_values(["Telepules"])

    cdfs = []
    probabilities = []
    for seed in seeds:
        test = LodigeTest(
            df["ld_" + party_name],
            df["Telepules"],
            bottom_n,
            iterations,
            seed,
            pe_iterations,
            pe_seed,
            quiet=False
        )

        """ TODO: move the seeder loop into the reused module? would make sense
                  only that it's a generic bootstrapping thing
        """
        # slightly redundant:
        probabilities.append(test.p)
        cdfs.append(test.cdf)

    # each test should say the same anyway, pick the last
    actual_likelihood = test.likelihood
    print("Actual (mean) likelihood:", actual_likelihood)
    print("Seeds:", seeds)
    print("Likelihood probabilities:", probabilities)

    sample = np.concatenate([np.array(cdf.sample) for cdf in cdfs])
    save_results(actual_likelihood, probabilities, sample)


def plot_summary():
    actual_likelihood, probabilities, sample = load_results()
    finite_sample = sample[~np.isinf(sample )]
    plot_entropy_distribution(
        actual_likelihood,
        np.mean(probabilities),
        finite_sample,
        is_quiet=is_quiet()
    )


if __name__ == "__main__":
    # min_wards=8
    # run_simulation(bottom_n=20)  # 1k: 0.021, 10k: 0.0236
    # run_simulation(bottom_n=20, seed=1235)  # 1k: 0.041
    # run_simulation(bottom_n=20, seed=1236, iterations=50)  # 1k: 0.03, 10k: 0.0271
    # run_simulation(bottom_n=50) # 100: 0.02, 500: 0.036, 1k: 0.036
    # run_simulation(bottom_n=100) # 100: 0.04, 500: 0.084, 1k: 0.076
    # run_simulation(bottom_n=10, iterations=100) # 100: 0.03, 1k: 0.032
    # run_simulation(bottom_n=15, iterations=100) # 100: 0.01, 1k: 0.027, 2k: 0.0295
    # run_simulation(bottom_n=20, iterations=1000, seeds=[1234],
    #                party_name="Ervenyes")
    run_simulation(bottom_n=20,
                   iterations=5000,
                   seeds=[1234, 1235, 1236, 1237],
                   party_name="Fidesz")

    # min_wards=4
    # run_simulation(bottom_n=200, iterations=1000)  # 1000: 0.308
    # run_simulation(bottom_n=100, iterations=1000)  # 1000: 0.229
    # run_simulation(bottom_n=10, iterations=1000)  # 1000: 0.09
    # run_simulation(bottom_n=20, iterations=1000)  # 1000: 0.127

"""
Fun:
import matplotlib.pyplot as plt


plt.hist(pd.concat([df.ld_Jobbik, df.ld_DK, df.ld_LMP, df.ld_MKKP, df.ld_MSZP, df.ld_Momentum]))

for d in range(9, 7, -1):
    plt.hist(df.Fidesz[df.ld_Fidesz == d], bins=30, alpha=0.5)

for d in [9, 7]:
     plt.hist(df.Fidesz[df.ld_Fidesz == d], bins=np.arange(0, 300, 30), alpha=0.5)
     plt.show()

plt.hist(pd.concat([df[df.Ervenyes > 1000].ld_Fidesz, df[df.Ervenyes > 1000].ld_Jobbik]), bins=range(11))
"""
