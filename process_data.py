import runpy


MODULE_NAMES = [
    "app5_ent_in_top",
    "explore_2018",
    "app6_comparative",
    "app7_fidesz_vs_jobbik",
    "app8_prob_of_twin_digits",
    "app9_overall_log_likelihood",
    "app14_digit_correlations_Hun",
]


def run_modules():
    for mod_name in MODULE_NAMES:
        print("Running", mod_name)
        runpy.run_module(mod_name, run_name='__main__')


if __name__ == "__main__":
    run_modules()
