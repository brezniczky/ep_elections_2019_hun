from PL.preprocessing import (get_data_sheets, merge_lista_results,
                              MergedDataInfo)
from drdigit.digit_entropy_distribution import LodigeTest
from drdigit.digit_filtering import get_feasible_rows, get_feasible_groups
from drdigit.scoring import get_group_scores
from drdigit.fingerprint_plots import (
    plot_fingerprint, plot_animated_fingerprints
)
import numpy as np


"""
Run the script with LL_ITERATIONS set to 5000 to get << 10% probabilities.
"""

SEED = 1234
LL_ITERATIONS = 1000
FINGERPRINT_DIR = "fingerprints_Poland"

SAMPLE_RATIO = None  # for a bit of robustness assessment, set to e.g. 0.95


# Per row checks yield a considerable statistic

# raise the min req. to
# 200 for a 5.1%
# or 300 and get 2.6%
# on lista 4
# clear sign of tampering
# p-value 0.0344  with 5k  iterations
# while lista 3 gets p-value 0.7078


# of course IF there was to be any cheating - you would definitely want to waste
# your cheating capacity where it is worth it or it is needed
# cities are generally more liberal and due to "closer proximity" of higher
# education (in the communication links sense), resistant to propaganda
# then they are also high leverage due to their size
# so ... each ward - counted by probably the same number of people - controls
# more votes, gives a better ROI in case of ... embedding/blackmailing vote
# counters? hacking?


# TODO:
# however, it would be worthwhile to restrict to the 300+ lista 4 votes, 100+
# lista 4 votes, and examine lista 4 probability - likely those were dragged
# down


def _apply_bootstrap(feasible_lista):
    """ Utility for a bit of bootstrapping based robustness assessment. """
    if SAMPLE_RATIO is not None:
        # ended up giving in to bootstrap style (i.e. with replacement)
        indexes = sorted(np.random.choice(range(len(feasible_lista)),
                                          int(len(feasible_lista) *
                                              SAMPLE_RATIO),
                                          replace=True))
        feasible_lista = feasible_lista.iloc[indexes]
    return feasible_lista


def check_overall_entropy_values_per_row(merged):
    """ Here we find that a 'per row' filtered lista 4 becomes suspicious. """
    np.random.seed(SEED)

    print("Relaxed (by row municipality filtering) entropy tests")
    # most "lista"s cannot be tested in this way since their values are too low
    for lista_index in [3, 4]:  # [1, 2, 3, 4, 5, 6, 7]:
        print("Testing lista %d ..." % lista_index)
        row_feasible_lista = merged.iloc[:, [0, lista_index]]
        row_feasible_lista = get_feasible_rows(row_feasible_lista, 300, [1])

        row_feasible_lista = _apply_bootstrap(row_feasible_lista)

        if len(row_feasible_lista) > 0:
            print("found %d feasible values in %d municipalities" %
                  (len(row_feasible_lista),
                   len(set(row_feasible_lista.iloc[:, 0]))))

            test_lista = LodigeTest(
                digits=row_feasible_lista.iloc[:, 1] % 10,
                group_ids=row_feasible_lista.iloc[:, 0],
                bottom_n=20,
                ll_iterations=LL_ITERATIONS
            )
            print("likelihood", test_lista.likelihood)
            print("p-value", test_lista.p)
        else:
            print("No feasible wards were found.")
        print()


def check_overall_entropy_values_per_municipality(merged):
    # This gave good, uninteresting p-values (>0.8)

    np.random.seed(1234)

    # let's take a look at lista 3 and 4 with the stricter restriction
    print("Stricter (\"municipality ~ city\") entropy tests")
    for lista_index in [3, 4]:
        feasible_settlements = get_feasible_groups(
            merged, 8, 50,
            value_colname=merged.columns[lista_index],
            group_colname=merged.columns[0]
        )
        print("%d feasible settlements were identified for lista %d" %
              (len(feasible_settlements), lista_index))
        feasible_settlements = _apply_bootstrap(feasible_settlements)
        city_feasible_lista = \
            merged[merged.iloc[:, 0].isin(feasible_settlements)]
        print("These involve %d wards" % len(city_feasible_lista))
        test = LodigeTest(
            city_feasible_lista.iloc[:, lista_index] % 10,
            city_feasible_lista.iloc[:, 0],
            bottom_n=20,
            ll_iterations=LL_ITERATIONS
        )
        likelihood = test.likelihood
        print("likelihood:" % likelihood)
        p = test.p
        print("p-value: %.2f" % p)


def check_ranking(merged, info):
    # TODO: should possibly row filter before this
    feasible_df = get_feasible_rows(
        merged,
        100,
        [list(merged.columns).index(info.get_lista_column(4))]
    )

    scores = get_group_scores(feasible_df[info.area_column],
                              feasible_df[info.get_lista_column(4)].values % 10,
                              overhearing_base_columns=[
                                  feasible_df[info.valid_votes_column].values % 10,
                                  feasible_df[info.get_lista_column(4)].values % 10,
                              ],
                              overhearing_indep_columns=[
                                  feasible_df[info.get_lista_column(3)].values % 10,
                              ],
                              )

    scores.sort_values(inplace=True)
    return scores


def print_top_k(merged, info, ranking, k):
    printed_cols = [
        info.area_column,
        info.nr_of_registered_voters_column,
        info.valid_votes_column,
        info.get_lista_column(3),
        info.get_lista_column(4),
    ]
    display_colnames = ["area", "registered", "valid_votes",
                        "lista_3", "lista_4"]
    for i, area_code in zip(range(k), ranking.index[:k]):
        print("\n#%d" % (i + 1))
        printed_df = merged[merged[info.area_column] == area_code][printed_cols]
        printed_df.columns = display_colnames
        print(printed_df)


def plot_PL_fingerprint(merged, info, areas, group_desc, lista_index):
    act_df = merged[merged[info.area_column].isin(areas)]
    plot_fingerprint(
        party_votes=act_df[info.get_lista_column(lista_index)],
        valid_votes=act_df[info.valid_votes_column],
        registered_voters=act_df[info.nr_of_registered_voters_column],
        title="Poland %s, 2019 EP, lista %d" %
              (group_desc, lista_index),
        fingerprint_dir=FINGERPRINT_DIR,
        filename="%s lista %d.png" % (group_desc, lista_index),
        quiet=True,
    )


def plot_fingerprints(merged, info: MergedDataInfo, ranking):
    n = len(ranking)

    plot_params = [
        [ranking.index[:90], "1-90"],
        [ranking.index[91:180], "91-180"],
        [ranking.index[:300], "1-300"],
        [ranking.index[301:600], "301-600"],
        [ranking.index[:int(n / 8)], "top 12.5%"],
        [ranking.index[int(n / 8):], "top 12.5% excluded"],
        [ranking.index[:int(n / 4)], "top 25%"],
        [ranking.index[int(n / 4):], "top 25% excluded"],
        [ranking.index[:int(n / 2)], "top 50%"],
        [ranking.index[int(n / 2):], "top 50% excluded"],
    ]

    for areas, group_desc in plot_params:
        for lista_index in [1, 2, 3, 4]:
            plot_PL_fingerprint(merged, info, areas, group_desc, lista_index)


def plot_animated_fingerprint_pairs(merged, info: MergedDataInfo, ranking):
    n = len(ranking)

    for top_perc in [25, 33, 50]:
        n_top = int(n * top_perc / 100)
        n_transl = n - n_top
        n_frames = 20

        frame_inclusions = []
        frame_title_exts = []
        for k in range(n_frames):
            s = int(n_transl * k / (n_frames - 1))
            act_areas = ranking.index[s:(s + n_top)]
            is_kth = merged[info.area_column].isin(act_areas)
            frame_inclusions.append(is_kth)
            frame_title_exts.append("%.2d" % k)

        # pause a bit at the end by repeating the last frame
        frame_inclusions.append(is_kth)
        frame_title_exts.append("  ")
        frame_inclusions.append(is_kth)
        frame_title_exts.append("  ")
        frame_inclusions.append(is_kth)
        frame_title_exts.append("  ")
        frame_inclusions.append(is_kth)
        frame_title_exts.append("  ")

        # fast rewind (2x SPEED SAME PRICE)
        for k in range(0, n_frames, 2):
            print("append ", k)
            frame_inclusions.append(frame_inclusions[n_frames - 1 - k])
            frame_title_exts.append("  ")

        for lista_index in [3, 4]:
            plot_animated_fingerprints(
                merged[info.get_lista_column(lista_index)],
                merged[info.valid_votes_column],
                merged[info.nr_of_registered_voters_column],
                frame_inclusions,
                "lista %d top %d%% to bottom %d%% suspects" %
                    (lista_index, top_perc, top_perc),
                "lista %d top %d to bottom %d suspects.gif" %
                    (lista_index, top_perc, top_perc),
                fingerprint_dir=FINGERPRINT_DIR,
                quiet=True,
                interval=50,
                frame_title_exts=frame_title_exts,
            )


def process_data(do_slow_things=True):
    # TODO: ... a proper random seeding strategy ...
    # the seed mainly ensures the fuzzy vote limit based selection
    np.random.seed(1234)

    dfs = get_data_sheets()

    merged, info = merge_lista_results(dfs, return_overview_cols=True)

    if do_slow_things:
        check_overall_entropy_values_per_row(merged)
        check_overall_entropy_values_per_municipality(merged)

    ranking = check_ranking(merged, info)
    print_top_k(merged, info, ranking, 20)
    plot_fingerprints(merged, info, ranking)
    plot_animated_fingerprint_pairs(merged, info, ranking)
    return dfs, ranking


if __name__ == "__main__":
    dfs, ranking = process_data()
