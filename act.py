# suspect3_limit = 2.4 - digit_sum_extr["sum"] * (2.5 - 1.4) / 140
# suspect3_intercept = 2.2
# suspect3_limit = suspect3_intercept - digit_sum_extr["sum"] * (suspect3_intercept - 1.3) / 140


# what are the odds that all of them deviate, exactly ...?
#

"""
Can't interpret this, but looks interesting.
"""
# plt.hist(df.ld_Fidesz[(df.LMP > 20)], bins=10)
# plt.show()



# Plot this:
# max_to_mean ratio against the number of votes in town
plt.scatter(digit_sum_extr["sum"], digit_sum_extr.max_to_mean, alpha=0.5)
plt.scatter(suspects2["sum"], suspects2.max_to_mean, alpha=0.5)
plt.show()


# try this: above (0, 2.3) -- (140, 0)
# i.e. 2.3 - 140 / 2.3 * "sum"


# A more intelligent filtering:
# (sure it's some CLT sqrt limit that should generally be designed, but
# here I just picked a line by the eye)





"""
What is the likeliness of getting 62 partial overlaps in 107 values?
"""

# TODO: sort order
# TODO: a faster approach can be to prove that this
#       correlates with the quasi-geographical order
#       just put them in a random order - and show
#       that it's much worse in (I guess) 90% of the cases
#       95 would be better though ...
# TODO: is the current order correct by the way? (or is it incorrectly sorted?)
# TODO: occurrance is occurrence
# TODO: prev digit = act digit test over the full dataset - why not?


def get_prob_of_dev_33_in_64(df_to_copy, expected_len):
    df = df_to_copy.copy()
    # there's a NaN-ful row, exclude that
    df = df.loc[df.Megye.apply(type) == str]

    # gave a 4.0 percent chance - with 1000 iterations
    # (TODO: despite the seed I may have seen inconsistent results?)
    np.random.seed(4321)

    def as_index(values):
        # ensure it is ordered for reproducibility
        # (sets have a slightly random order)
        value_set_list = sorted(list(set(values)))
        value_index_dict = dict(
          zip(list(value_set_list), range(len(value_set_list)))
        )
        return [value_index_dict[v] for v in values]

    # combining the typical aggregation key apparently gets
    # things slightly faster
    df["Megyepules"] = df.apply( \
      lambda x: x.Telepules + "|" + x.Megye,
      axis=1
    )
    df.Megyepules = as_index(df.Megyepules)

    hits = 0
    bulls = 0
    misses = 0
    start = datetime.now()

    for i in range(1000):
        df.ld_Fidesz = (
            np.random.choice(last_digit_pop,
                             len(df))
        )

        # repeat the full process
        town_groups = \
            df[
                ["Megyepules", "ld_Fidesz"]
            ].groupby(["Megyepules", "ld_Fidesz"])

        county_town_digit_sums = town_groups.aggregate(len)

        # max_to_min, min are not used - skip them for speed
        digit_sum_extr = county_town_digit_sums \
                         .groupby(["Megyepules"]) \
                         .aggregate([max, 'mean',
                                     sum, lucky_nr, lucky_nr2])
        # digit_sum_extr["max_to_min"] = digit_sum_extr["max"] / digit_sum_extr["min"]
        digit_sum_extr["max_to_mean"] = digit_sum_extr["max"] / digit_sum_extr["mean"]

        # need to find 107-element sequences, not so easy, need a little adapting
        k = 0

        while True:
            suspect2_limit = 2.5 - digit_sum_extr["sum"] * (2.5 - 1.4) / 140 - k / 200

            is_suspect2 = digit_sum_extr.max_to_mean > suspect2_limit
            if sum(is_suspect2) >= expected_len:
                break
            k += 1

        if sum(is_suspect2) > expected_len:
            lose = np.random.choice(np.where(is_suspect2)[0],
                                    sum(is_suspect2) - expected_len)
            is_suspect2[lose] = False

        suspects2 = digit_sum_extr.loc[is_suspect2]

        counter = 0

        for j in range(len(suspects2) - 1):
            dig1 = suspects2.iloc[j][["lucky_nr", "lucky_nr2"]]
            dig2 = suspects2.iloc[j + 1][["lucky_nr", "lucky_nr2"]]
            if not np.isnan(dig1[1]) and not np.isnan(dig2[1]):
                intersection = set(dig1).intersection(set(dig2))
                counter += len(intersection)
            else:
                print("nan found")

        if counter >= 33:
            print("hit", counter)
            hits += 1
        else:
            print("missed")
            misses += 1
            # # two zeroes
            # if (sorted_counts[0] +
            #     sorted_counts[1] +
            #     sorted_counts[2] >= 27 / 63 * sum(sorted_counts)):

            #         bulls += 1

        if i % 5 == 0:
            print(datetime.now() - start, i, hits, misses) #, bulls)

    print("hits:", hits, "misses:", misses)  # "bull hits:", bulls,
    return hits / (hits + misses)   #  , bulls / (hits + misses)


print(get_prob_of_dev_33_in_64(df, len(suspects2)))


plt.hist(df.ld_Fidesz[(df.LMP > 20)], bins=10); plt.show()

exit(1)




"""
Another oddity is that while the digit "9" and "0" are in a way
adjacent - assuming larger values become less frequent, 0 comes
"after" 9, and its frequency should be lower than that of "9"s

However, what we see in case of one of the big "losers" of the
election is that the relationship is quite the opposite.
"""
plt.hist(df.ld_LMP[df.Fidesz > 50]); plt.show()





all_digits = list(range(10))
hits = 0
np.random.seed(1111)
for i in range(10000):
    d1 = np.random.choice(range(10))
    d2 = np.random.choice(all_digits[:d1] + all_digits[d1 + 1:])

    e1 = np.random.choice(range(10))
    d2 = np.random.choice(all_digits[:e1] + all_digits[e1 + 1:])

    if (d1 in [e1, e2]) or (d2 in [e1, e2]):
        hits += 1

print(hits)



suspect3_limit_lo = 13
suspect3_limit_hi = 1000

scaled = digit_sum_extr.max_to_mean * (digit_sum_extr["sum"] ** 0.5)
is_suspect3 = ((suspect3_limit_hi >= scaled) & (scaled >= suspect3_limit_lo))
suspects3 = digit_sum_extr.loc[is_suspect3]
print(len(suspects3))


plt.hist(suspects3.lucky_nr)
plt.show()


plt.scatter(digit_sum_extr["sum"], digit_sum_extr.max_to_mean * (digit_sum_extr["sum"] ** 0.5), alpha=0.5)
plt.scatter(suspects3["sum"], suspects3.max_to_mean* (suspects3["sum"] ** 0.5), alpha=0.5)
plt.show()

print("As we see it, it's even more striking how the\n"
      "distribution appears to unveil more and more evidence\n"
      "of tampering.\n")

print("24 of 32 numbers are from the 5 most frequent ones.\n"
      "what is the probability?")


# stop it now


exit(1)


def test_full_process_with_remodelled_data_susp3(df_to_copy, expected_len):
    df = df_to_copy.copy()
    # there's a NaN-ful row, exclude that
    df = df.loc[df.Megye.apply(type) == str]

    # gave a 4.0 percent chance - with 1000 iterations
    # (TODO: despite the seed I may have seen inconsistent results?)
    np.random.seed(4321)

    def as_index(values):
        # ensure it is ordered for reproducibility
        # (sets have a slightly random order)
        value_set_list = sorted(list(set(values)))
        value_index_dict = dict(
          zip(list(value_set_list), range(len(value_set_list)))
        )
        return [value_index_dict[v] for v in values]

    # combining the typical aggregation key apparently gets
    # things slightly faster
    df["Megyepules"] = df.apply( \
      lambda x: x.Telepules + "|" + x.Megye,
      axis=1
    )
    df.Megyepules = as_index(df.Megyepules)

    hits = 0
    bulls = 0
    misses = 0
    start = datetime.now()

    for i in range(1000):
        df.ld_Fidesz = (
            np.random.choice(last_digit_pop,
                             len(df))
        )

        # repeat the full process
        town_groups = \
            df[
                ["Megyepules", "ld_Fidesz"]
            ].groupby(["Megyepules", "ld_Fidesz"])

        county_town_digit_sums = town_groups.aggregate(len)

        # max_to_min, min are not used - skip them for speed
        digit_sum_extr = county_town_digit_sums \
                         .groupby(["Megyepules"]) \
                         .aggregate([max, 'mean',
                                     # min,
                                     sum, lucky_nr])
        # digit_sum_extr["max_to_min"] = digit_sum_extr["max"] / digit_sum_extr["min"]
        digit_sum_extr["max_to_mean"] = digit_sum_extr["max"] / digit_sum_extr["mean"]

        # need to find 27-element sequences, not so easy, need a little adapting
        k = 0

        while True:
            suspect3_limit = 2.5 - digit_sum_extr["sum"] * (2.5 - 1.4) / 140 - k / 200

            is_suspect3 = digit_sum_extr.max_to_mean > suspect3_limit
            if sum(is_suspect3) >= expected_len:
                break
            k += 1

        if sum(is_suspect3) > expected_len:
            lose = np.random.choice(np.where(is_suspect3)[0],
                                    sum(is_suspect3) - expected_len)
            is_suspect3[lose] = False

        suspects3 = digit_sum_extr.loc[is_suspect3]

        digits_and_counts = \
          np.unique(suspects3.lucky_nr, return_counts=True)
        sorted_counts = sorted(digits_and_counts[1], reverse=True)

        if sum(sorted_counts[:5]) >= 24:
            print(sorted_counts)
            hits += 1
        else:
            misses += 1
            # # two zeroes
            # if (sorted_counts[0] +
            #     sorted_counts[1] +
            #     sorted_counts[2] >= 27 / 63 * sum(sorted_counts)):

            #         bulls += 1

        if i % 5 == 0:
            print(datetime.now() - start, i, hits, misses) #, bulls)

    print("hits:", hits, "misses:", misses)  # "bull hits:", bulls,
    return hits / (hits + misses)   #  , bulls / (hits + misses)


test_full_process_with_remodelled_data_susp3(df, expected_len=len(suspects3))


