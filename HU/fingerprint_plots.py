import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from drdigit import plot_overlaid_fingerprints


def plot_histogram2d(d_2_1, d_1_2, binx, biny, show=True, filename=None):
    # plot a numpy histogram2d result via matplotlib

    # plt.pcolormesh(binx, biny, d_1_2, alpha=0.5)

    # couldn't even try out the , shading="gouraud" due to a
    # "Dimensions of C (99, 99) are incompatible with X (100) and/or Y (100);
    # see help(pcolormesh)" probably due to
    # https://github.com/matplotlib/matplotlib/issues/8422
    #
    # alpha=0.5 otherwise looks horrible unfortunately

    # cmap1 = plt.get_cmap('PiYG')
    div = np.maximum(np.max(d_2_1), np.max(d_1_2)) * 1.3

    cmap1 = plt.get_cmap('copper')
    cmap2 = plt.get_cmap('bone')

    plt.pcolormesh(binx, biny, d_1_2, cmap=cmap1, norm=Normalize(vmax=div))
    d_2_1 = np.ma.masked_array( d_2_1, d_2_1 == 0)
    plt.pcolormesh(binx, biny, d_2_1, cmap=cmap2, norm=Normalize(vmax=div))

    if filename:
        plt.savefig(filename)

    if show:
        plt.show()


def get_diffs(x1, y1, w1, x2, y2, w2):
    # TODO: zooming

    # 1. histogram gets x arr, y arr, weights
    #    returns values (nx x ny), x bin edges (n+1), y bin edges (n+1)
    #    will spec the bins on input so that they conform on return for 1 and 2
    bins = [np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)]

    v1, x1, y1 = np.histogram2d(x1, y1, bins=bins, weights=w1)
    v2, x2, y2 = np.histogram2d(x2, y2, bins=bins, weights=w2)

    assert all(x1 == x2) and all(y1 == y2)

    # 2. calc. intersection (inert area)
    def is_(x):
        # median or mean as a fallback to act as a threshold of
        # 'surely sufficiently convincingly included'
        mx = np.median(x)
        if mx == 0:
            mx = np.mean(x)
        # 0..1 value for AND-style multiplicability
        return np.minimum(x / mx, 1)

    # operate on slightly fuzzy logical masks here
    is_intersection = is_(v1) * is_(v2)
    is_v1_only = is_(v1) * (1 - is_intersection)
    is_v2_only = is_(v2) * (1 - is_intersection)

    # the product is still in 0..1 and can be used to slice the A or B-specific
    return v1 * is_v1_only, v2 * is_v2_only, x1, y1


def print_fingerprint_diff_stats(df, party, top_municipalities,
                                 bottom_municipalities):
    df_top = df[
        df.Telepules.isin(top_municipalities)
    ]
    df_bottom = df[
        df.Telepules.isin(bottom_municipalities)
    ]

    s1 = sum(df_top[party])
    s2 = sum(df_bottom[party])
    print("votes won when not susp", s2, "susp", s1, "ratio", s1 / s2)
    df1 = df_top
    df2 = df_bottom
    print("mean valid votes:",
          np.mean(df1["Ervenyes"]),
          np.mean(df2["Ervenyes"]))
    print("total valid votes ratio \"achieved:",
          np.sum(df1["Ervenyes"]) /
          np.sum(df2["Ervenyes"]))
    print("vote ratio in susp. areas %.2f" % (s1 / sum(df1["Ervenyes"]) * 100))
    print("additional votes perc. in more susp. areas %.2f %%" %
          ((s1 - s2) / (sum(df1["Ervenyes"]) - sum(df2["Ervenyes"])) * 100))
    # have to go CLT
    #
    # std_indiv / sqrt(n_sampl)   ~   sd of the mean
    # std_indiv * sqrt(n_sampl)   ~   sd of the sum
    # sd of the sum over the sum  ~   rel. uncertainty in nr. of votes
    print("rel. sd ratio (< 1 for suspect targeted manipulation):",
          (np.std(df1[party] * (len(df1) ** 0.5)) / np.sum(df1[party])) /
          (np.std(df2[party] * (len(df2) ** 0.5)) / np.sum(df2[party]))
          )


def plot_fingerprint_diff(df, party, top_municipalities, bottom_municipalities,
                          title="", show=True,
                          filename=None, fingerprint_dir=None):

    # # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/quadmesh_demo.html
    # # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html
    df1 = df[
        df.Telepules.isin(top_municipalities)
    ]
    df2 = df[
        df.Telepules.isin(bottom_municipalities)
    ]

    if fingerprint_dir is not None and filename is not None:
        filename = os.path.join(fingerprint_dir, filename)

    ans = plot_overlaid_fingerprints(
        party_votes=[df1[party], df2[party]],
        valid_votes=[df1["Ervenyes"], df2["Ervenyes"]],
        registered_voters=[df1["Nevjegyzekben"], df2["Nevjegyzekben"]],
        quiet=not show,
        filename=filename,
        title=title,
        legend_strings=["most", "least"]
    )

    return ans


def plot_comparative(values1, values2):

    col1 = np.array([0.3, 0.5, 0.9])
    col2 = np.array([0.8, 0.5, 0.3])

    def colormap_from_intensity(values, col):
        return np.array([
            [col * value for value in row]
            for row in values
        ])

    map = (colormap_from_intensity(values1, col1) +
           colormap_from_intensity(values2, col2)) / 2

    plt.imshow(map)
    plt.show()


if __name__ == "__main__":
    import numpy.random as rnd
    rnd.seed(1234)
    values1 = (rnd.uniform(0, 1, 30 * 30) *
               np.arange(0, 1, 1 / 900)).reshape((30, 30))
    values2 = (rnd.uniform(0, 1, 30 * 30) *
               np.arange(1, 0, -1 / 900)).reshape((30, 30))

    plot_comparative(values1, values2)
