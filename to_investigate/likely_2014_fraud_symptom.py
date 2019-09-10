from HU.cleaning import get_2014_cleaned_data, get_2010_cleaned_data
from HU.app16_fingerprint_plots import plot_municipality

plot_municipality("Miskolc", 'Fidesz', 2014, highlight_last_digit=0)

# some ward indexes from the right hand side of chart
point_indexes = [116, 4, 56, 60, 55, 61, 83, 26, 27, 60, 14, 17, 158, 154, 137, 135, 86]

df = get_2014_cleaned_data()
print(df[(df.Telepules == "Miskolc") & (df.Szavazokor.astype(int).isin(point_indexes))]["MSZP-Együtt-DK-PM-MLP"])


"""
Look at those last digits ...

In [322]: df[(df.Telepules == "Miskolc") & (df.Szavazokor.astype(int).isin(point_indexes))]["MSZP-Együtt-DK-PM-MLP"]
Out[322]:
1849    217
1853    216
2025    235
2245    218
2718    138
2719    215
2723    178
2800    208
2801    188
2810    226
3553    193
3614    251
3617    170
3628    397
4892    153
4902    435
Name: MSZP-Együtt-DK-PM-MLP, dtype: int64

"""

plot_municipality("Miskolc", 'Fidesz', 2014, highlight_last_digit=0)

# some ward indexes from the right hand side of chart
# point_indexes = [28, 91, 32, 31, 137, 11, 25, 61, 92, 27, 134, 157, 46, 160, 50, 30, 24, 85]

point_indexes = [28, 91, 32, 31, 61, 92, 27, 67, 25, 85, 24, 30, 50, 152, 137, 158, 11, 160, 46, 157]


df = get_2010_cleaned_data()

print(df[(df.Telepules == "Miskolc") & (df.Szavazokor.astype(int).isin(point_indexes))]["Fidesz-KDNP"])

"""
Looks quite distastefully redundant in terms of last digit
2317    213.0
2330    359.0
2331    355.0
2333    246.0
2334    294.0
2336    329.0
2337    270.0
2338    334.0
2352    241.0
2356    208.0
2367    261.0
2373    240.0
2391    215.0
2397    487.0
2398    416.0
2443    224.0
2458    291.0
2463    188.0
2464    207.0
2466    196.0
Name: Fidesz-KDNP, dtype: float64
"""

# Budapest XXII. district. An extremely beautiful demonstration
