from PL.preprocessing import get_big_cleaned_data, merge_lista_results
from drdigit.digit_entropy_distribution import LodigeTest
from drdigit.digit_filtering import get_feasible_rows, get_feasible_groups
from drdigit.scoring import get_group_scores
import numpy as np


SEED = 1236
LL_ITERATIONS = 1000


_ranking = None

# Per row checks yield a considerable

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


def check_overall_entropy_values_per_row(merged):
    np.random.seed(SEED)

    print("Relaxed (by row municipality filtering) entropy tests")
    # # most "lista"s cannot be tested in this way since their values are too low
    for lista_index in [3, 4]:  # [1, 2, 3, 4, 5, 6, 7]:
        print("Testing lista %d ..." % lista_index)
        row_feasible_lista = merged.iloc[:, [0, lista_index]]
        row_feasible_lista = get_feasible_rows(row_feasible_lista, 300, [1])

        indexes = sorted(np.random.choice(range(len(row_feasible_lista)),
                                          int(len(row_feasible_lista) * 0.95),
                                          replace=False))
        row_feasible_lista = row_feasible_lista.iloc[indexes]
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
            # gives >50% and 14.4% for lista 3 and 4, respectively
            # basically good p-values except for the winning lista
        else:
            print("No feasible wards were found.")
        print()


def check_overall_entropy_values_per_municipality(merged):
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


def check_ranking(merged):
    # TODO: should possibly row filter before this
    row_feasible_lista = merged.iloc[:, [0, 4]]
    row_feasible_lista = get_feasible_rows(row_feasible_lista, 100, [1])
    scores = get_group_scores(row_feasible_lista.iloc[:, 0],
                              row_feasible_lista.iloc[:, 1])

    scores.sort_values(inplace=True)
    return scores


def process_data():
    dfs = get_big_cleaned_data()
    merged = merge_lista_results(
        dfs,
        lista_idxs_to_exclude=[8, 9, 10]
    )
    check_overall_entropy_values_per_row(merged)
    check_overall_entropy_values_per_municipality(merged)
    global _ranking  # for debugging stuff
    _ranking = check_ranking(merged)


if __name__ == "__main__":
    process_data()
