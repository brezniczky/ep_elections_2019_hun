from PL.preprocessing import (get_data_sheets, merge_lista_results,
                              MergedDataInfo)
from drdigit import (LodigeTest,
                     filter_df,
                     get_group_scores, plot_fingerprint,
                     plot_animated_fingerprints,
                     plot_entropy_distribution)
import numpy as np
from arguments import is_quiet, is_quick, get_output_dir
import os


"""
Run the script with LL_ITERATIONS set to 5000 to get << 10% probabilities.
"""

SEED = 1234
LL_ITERATIONS = 50000 if not is_quick() else 1000
FINGERPRINT_DIR = os.path.join(get_output_dir(), "fingerprints_Poland")

SAMPLE_RATIO = None  # for a bit of robustness assessment, set to e.g. 0.95


# Per row checks yield a considerable statistic

# min req. values now yield changed probabilities
# (the underlying MC was changed for performance benefits)
# now using 50k LL iterations:

#        Lista 3   Lista 4
# 100 -   71.66%     6.5%  .
# 200 -   62.93%     0.29% **
# 300 -   87.44%     6.92% .

# of course IF there was to be any cheating - you would definitely want to waste
# your cheating capacity where it is worth it or it is needed
# cities are generally more liberal and due to "closer proximity" of higher
# education (in the communication links sense), resistant to propaganda
# then they are also high leverage due to their size
# so ... each ward - counted by probably the same number of people - controls
# more votes, gives a better ROI in case of ... embedding/blackmailing vote
# counters? hacking?


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
    """ Here we find that a 'per row' filtered Lista 4 becomes suspicious. """
    np.random.seed(SEED)

    print("Relaxed (by row municipality filtering) entropy tests")
    # most "lista"s cannot be tested in this way since their values are too low
    for lista_index in [3, 4]:  # [1, 2, 3, 4, 5, 6, 7]:
        print("Testing Lista %d ..." % lista_index)
        row_feasible_lista = merged.iloc[:, [0, lista_index]]
        row_feasible_lista = filter_df(
            row_feasible_lista, min_value=100,
            value_columns=[merged.columns[lista_index]]
        )
        row_feasible_lista = _apply_bootstrap(row_feasible_lista)

        if len(row_feasible_lista) > 0:
            print("found %d feasible values in %d municipalities" %
                  (len(row_feasible_lista),
                   len(set(row_feasible_lista.iloc[:, 0]))))

            test_lista = LodigeTest(
                digits=row_feasible_lista.iloc[:, 1] % 10,
                group_ids=row_feasible_lista.iloc[:, 0],
                bottom_n=20,
                ll_iterations=LL_ITERATIONS,
                avoid_inf=True  # prevent Lista 4 -Inf LL
            )
            print("likelihood:", test_lista.likelihood)
            print("p-value: %.2f" % test_lista.p)
            if not is_quiet():
                plot_entropy_distribution(
                    test_lista.likelihood,
                    test_lista.p,
                    test_lista.cdf.sample,
                    title="Lista %d (selected per ward)" % lista_index
                )
        else:
            print("No feasible wards were found.")
        print()


def check_overall_entropy_values_per_municipality(merged):
    # This gave good, uninteresting p-values (>0.6)

    np.random.seed(1234)

    # let's take a look at lista 3 and 4 with the stricter restriction
    print("Stricter (\"municipality ~ city\") entropy tests")
    for lista_index in [3, 4]:
        city_feasible_lista = filter_df(
            merged,
            group_column=merged.columns[0], min_group_size=8,
            value_columns=[merged.columns[lista_index]], min_value=50,
        )
        feasible_settlements = set(city_feasible_lista.iloc[:, 0])
        feasible_settlements = _apply_bootstrap(feasible_settlements)
        print("%d feasible settlements were identified for Lista %d" %
              (len(feasible_settlements), lista_index))
        city_feasible_lista = city_feasible_lista[
            city_feasible_lista.iloc[:, 0].isin(feasible_settlements)
        ]
        print("These involve %d wards" % len(city_feasible_lista))
        test = LodigeTest(
            city_feasible_lista.iloc[:, lista_index] % 10,
            city_feasible_lista.iloc[:, 0],
            bottom_n=20,
            ll_iterations=LL_ITERATIONS,
            avoid_inf=True,  # Lista 4 has a nasty value in this scenario
        )
        print("likelihood:", test.likelihood)
        if not is_quiet():
            plot_entropy_distribution(
                test.likelihood,
                test.p,
                test.cdf.sample,
                title="Lista %d (selected per municipality)" % lista_index,
            )
        print("p-value: %.2f" % test.p)


def check_ranking(merged, info):
    feasible_df = filter_df(
        merged,
        min_value=100,
        value_columns=[info.get_lista_column(4)]
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


def plot_PL_fingerprint(merged, info, areas, group_desc, lista_index,
                        save: bool=False):
    act_df = merged[merged[info.area_column].isin(areas)]
    plot_fingerprint(
        party_votes=act_df[info.get_lista_column(lista_index)],
        valid_votes=act_df[info.valid_votes_column],
        registered_voters=act_df[info.nr_of_registered_voters_column],
        title="Poland %s, 2019 EP, Lista %d" %
              (group_desc, lista_index),
        fingerprint_dir=FINGERPRINT_DIR,
        filename="%s lista %d.png" % (group_desc, lista_index)
                 if save else None,
        quiet=is_quiet(),
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
    ] if not is_quick() else [
        [ranking.index[:int(n / 8)], "top 12.5%"],
        [ranking.index[int(n / 8):], "top 12.5% excluded"],
    ]

    lista_indexes = [1, 2, 3, 4] if not is_quick() else [
        3, 4
    ]

    for areas, group_desc in plot_params:
        for lista_index in [1, 2, 3, 4]:
            plot_PL_fingerprint(merged, info, areas, group_desc, lista_index,
                                save=True)


def plot_animated_fingerprint_pairs(merged, info: MergedDataInfo, ranking):
    n = len(ranking)

    top_percentages = [25, 33, 50] if not is_quick() else [25]
    n_frames = 20 if not is_quick else 2

    for top_perc in [25, 33, 50]:
        n_top = int(n * top_perc / 100)
        n_transl = n - n_top

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
