"""
Verifying feasible wards of the actual election data against
random uniform data based on the log likelihood distribution of
the n least likely candidates.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import get_preprocessed_data
from digit_entropy_distribution import get_entropy, prob_of_entr
from collections import OrderedDict
from app5_ent_in_top import plot_entropy_distribution_of


_DEFAULT_BOTTOM_N = 20
_DEFAULT_RANDOM_SEED = 1234
_DEFAULT_ITERATIONS = 1000
_DEFAULT_PE_RANDOM_SEED = 1234
_DEFAULT_PE_ITERATIONS = 12345


def get_slice_limits(settlement_values):
    values = np.unique(settlement_values)
    settlement_index = {
        name: i for name, i in zip(values, range(len(values)))
    }
    assert settlement_index != sorted(settlement_index), \
           "Settlement values must be in a sorted order!"

    indexes = pd.Series([settlement_index[s] for s in settlement_values])
    prev_indexes = pd.Series([-1] + list(indexes[:-1]))
    next_indexes = pd.Series(list(indexes[1:]) + [len(indexes)])
    starts = np.where(indexes != prev_indexes)[0]
    ends = np.where(indexes != next_indexes)[0] + 1
    return zip(starts, ends)


def get_log_likelihood(digits, slice_limits, bottom_n,
                       seed, iterations, towns=None):
    slices = [digits[a:b] for a, b in slice_limits]
    entropies = [get_entropy(s) for s in slices]
    probs = [prob_of_entr(len(s), e, seed, iterations)
            for s, e in zip(slices, entropies)]
    bottom_probs = sorted(probs)[:bottom_n]
    if towns is not None:
        print(towns[np.array(probs) <= max(bottom_probs)])
    l = sum(np.log(bottom_probs))
    return l


def get_likelihood_cdf(slice_limits, bottom_n,
                       seed=_DEFAULT_RANDOM_SEED,
                       iterations=_DEFAULT_ITERATIONS,  # voting simulation
                       pe_seed=_DEFAULT_RANDOM_SEED,
                       pe_iterations=_DEFAULT_ITERATIONS):  # entropy

    sample = []
    # end of the last slice is the ...
    n_settlements = slice_limits[-1][1]
    warnings = 0

    np.random.seed(seed)
    for i in range(iterations):
        digits = np.random.choice(range(10), n_settlements)
        sim_likelihood = get_log_likelihood(digits, slice_limits,
                                            bottom_n, pe_seed, pe_iterations)
        sample.append(sim_likelihood)
        if np.isinf(sim_likelihood):
            print("Warning! Infinite simulated likelihood - perhaps increase the p.e. iterations!")
            warnings += 1
        if i % 50 == 0:
            print(i, np.mean(np.array(sample)[~np.isinf(sample)]))

    values, counts = np.unique(sample, return_counts=True)
    total = sum(counts)
    counts = np.cumsum(counts)

    def cdf(l):
        """ Forgiving CDF: P(L <= L_actual) """
        idx = np.digitize(l, values) - 1
        if idx >= 0:
            return counts[idx] / total
        else:
            return 0
    print("There were", warnings, "warnings.")

    cdf.min = min(values[~np.isinf(values)])
    cdf.max = max(values[~np.isinf(values)])
    cdf.sample = sample

    return cdf


def get_feasible_settlements(df, min_n_wards, min_fidesz_votes):
    d = OrderedDict()
    agg = (
        df[["Telepules", "Fidesz"]]
        .groupby(["Telepules"])
        .aggregate(OrderedDict([("Telepules", len), ("Fidesz", min)]))
    )
    agg.columns = ["n_wards", "min_fidesz_votes"]
    agg = agg.reset_index()
    return agg.loc[(agg.n_wards >= min_n_wards) &
                   (agg.min_fidesz_votes >= min_fidesz_votes)]["Telepules"]


def save_results(actual_likelihood, probabilities, sample):
    df = pd.DataFrame(dict(names=["actual_likelihood"],
                           values=[actual_likelihood]))
    df.to_csv("app9_likelihood.csv", index=False)

    df = pd.DataFrame(dict(probability=probabilities))
    df.to_csv("app9_probabilities.csv", index=False)

    df = pd.DataFrame(dict(likelihood=sample))
    df.to_csv("app9_sample.csv", index=False)


def load_results():
    actual_likelihood = \
        pd.read_csv("app9_likelihood.csv").iloc[0]["values"]
    probabilities = \
        pd.read_csv("app9_probabilities.csv").probability
    sample = pd.read_csv("app9_sample.csv").likelihood

    return actual_likelihood, probabilities, sample


def run_simulation(bottom_n=_DEFAULT_BOTTOM_N,
                   seed=None,
                   seeds=None,
                   iterations=_DEFAULT_ITERATIONS,
                   pe_seed=_DEFAULT_PE_RANDOM_SEED,
                   pe_iterations=_DEFAULT_PE_ITERATIONS):
    if seeds is None:
        if seed is None:
            seeds = [_DEFAULT_RANDOM_SEED]
        else:
            seeds = [seed]
    else:
        assert seed is None, "Only seed or seeds should be specified not both"

    df = get_preprocessed_data()
    feasible_settlements = \
        get_feasible_settlements(df, min_n_wards=8, min_fidesz_votes=100)
    print("Found", len(feasible_settlements), "feasible settlements")
    df = df[df["Telepules"].isin(feasible_settlements)]
    df = df.sort_values(["Telepules"])

    slice_limits = list(get_slice_limits(df["Telepules"]))

    actual_likelihood = get_log_likelihood(
        df.ld_Fidesz.values,
        slice_limits,
        bottom_n,
        pe_seed, pe_iterations,
        towns=np.unique(df["Telepules"])
    )

    print("Actual likelihood:", actual_likelihood)
    # beware: this all hinges on a "by coincidence" independence
    # the seed is reset each time a new ward count is encountered by
    # the individual ward entropy cdf generator
    #
    # however, by the time we get to this point, all of them should
    # have been encountered at least once (the ward counts never change)
    # due to the actual entropy log likelihood probability calculations
    # so no more resets, "seeds" are taking control now:
    # (and yes, some reconsiderations are due in the longer run, if any)
    cdfs = []
    probabilities = []
    for seed in seeds:
        cdf = get_likelihood_cdf(slice_limits, bottom_n,
                                 seed, iterations,
                                 pe_seed, pe_iterations)
        cdfs.append(cdf)
        probabilities.append(cdf(actual_likelihood))

    actual_likelihood_prob = np.mean(probabilities)
    print("Actual (mean) likelihood prob:", cdf(actual_likelihood))
    print("Seeds:", seeds)
    print("Likelihood probabilities:", probabilities)

    sample = np.concatenate([np.array(cdf.sample) for cdf in cdfs])
    save_results(actual_likelihood, probabilities, sample)


def plot_summary():
    actual_likelihood, probabilities, sample = load_results()
    finite_sample = sample[~np.isinf(sample )]
    plot_entropy_distribution_of(
        actual_likelihood,
        probabilities,
        finite_sample,
    )


def test():
    res = list(get_slice_limits([1, 2, 2, 3, 3, 3]))
    assert (
        res ==
            [(0, 1),
             (1, 3),
             (3, 6)]
    )


if __name__ == "__main__":
    test()
    # min_wards=8
    # run_simulation(bottom_n=20)  # 1k: 0.021, 10k: 0.0236
    # run_simulation(bottom_n=20, seed=1235)  # 1k: 0.041
    # run_simulation(bottom_n=20, seed=1236, iterations=50)  # 1k: 0.03, 10k: 0.0271
    # run_simulation(bottom_n=50) # 100: 0.02, 500: 0.036, 1k: 0.036
    # run_simulation(bottom_n=100) # 100: 0.04, 500: 0.084, 1k: 0.076
    # run_simulation(bottom_n=10, iterations=100) # 100: 0.03, 1k: 0.032
    # run_simulation(bottom_n=15, iterations=100) # 100: 0.01, 1k: 0.027, 2k: 0.0295
    run_simulation(bottom_n=20, iterations=5000, seeds=[1234, 1235, 1236, 1237])

    # min_wards=4
    # run_simulation(bottom_n=200, iterations=1000)  # 1000: 0.308
    # run_simulation(bottom_n=100, iterations=1000)  # 1000: 0.229
    # run_simulation(bottom_n=10, iterations=1000)  # 1000: 0.09
    # run_simulation(bottom_n=20, iterations=1000)  # 1000: 0.127

"""
Fun:
plt.hist(pd.concat([df.ld_Jobbik, df.ld_DK, df.ld_LMP, df.ld_MKKP, df.ld_MSZP, df.ld_Momentum]))

for d in range(9, 7, -1):
    plt.hist(df.Fidesz[df.ld_Fidesz == d], bins=30, alpha=0.5)

for d in [9, 7]:
     plt.hist(df.Fidesz[df.ld_Fidesz == d], bins=np.arange(0, 300, 30), alpha=0.5)
     plt.show()

plt.hist(pd.concat([df[df.Ervenyes > 1000].ld_Fidesz, df[df.Ervenyes > 1000].ld_Jobbik]), bins=range(11))
"""
