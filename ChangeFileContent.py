import argparse
import os

import pandas as pd


def ChangeFileContent(store, prior, exp):
    if isinstance(store, str):
        store = pd.HDFStore(store, "a")
    if isinstance(prior, str):
        prior = pd.read_csv(prior, index_col=0)
    if isinstance(exp, str):
        exp = pd.read_csv(exp, index_col=0)

    config = store.get_storer("PriorAndConfig").attrs.my_attribute
    prior = prior.T
    prior[prior.columns.difference(["Type"])] = prior[
        prior.columns.difference(["Type"])
    ].astype("float")

    store["PriorAndConfig"] = prior
    store.get_storer("PriorAndConfig").attrs.my_attribute = config
    store["Exp_Y"] = exp.loc["Values"].astype("float")
    store["Exp_YErr"] = exp.loc["Errors"].astype("float")


if __name__ == "__main__":

    parser = argparser.ArgumentParser(
        description="This script changes the prior and experimental values of an already trained config file")
    parser.add_argument(
        "store",
        help="Location of the already existing config file")
    parser.add_argument("prior", help="Location of parameter priors")
    parser.add_argument("exp", help="Location of the experimental result")

    args = vars(parser.parse_args())
    ChangeFileContent(**args)
