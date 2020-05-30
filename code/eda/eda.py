"""
File: eda.py
Author: Alex Klapheke
Email: alexklapheke@gmail.com
Github: https://github.com/alexklapheke
Description: Tools for basic exploratory data analysis

Copyright © 2020 Alex Klapheke

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def summary(df):
    """Describe the columns of a data frame `df` with
    excerpted values, types, and summary statistics."""
    return pd.concat([
            df.head(5).T,
            pd.DataFrame({"type": df.dtypes}),
            pd.DataFrame({"nulls": df.isna().sum()}),
            df.describe().T
        ], axis=1)


def summary_group(df, col):
    """Describe the columns of a data frame `df`, grouped by column
    `col` with excerpted values, types, and summary statistics."""
    return pd.concat([
            df.groupby(col).first().T.add_prefix("head_"),
            df.groupby(col).dtypes.T.add_prefix("type_"),
            df.groupby(col).apply(lambda x: x.isna().sum()).T.add_prefix("nulls_"),
        ], axis=1)

def test_LINE(true, pred):
    """Test LINE assumptions for linear regression. Example usage:

        from sklearn.linear_model import LinearRegression

        # Generate data
        x = np.arange(100)
        np.random.shuffle(x)
        x = x + np.random.normal(size=100)
        y = x + np.random.normal(size=100) * 20
        X = x.reshape(-1, 1)

        # Fit data
        lr = LinearRegression().fit(X, y)

        # Show assumptions
        test_LINE(y, lr.predict(X))
    """

    # Set up graphs
    fig = plt.figure(figsize=(12, 8))

    # Define our residuals
    resids = true - pred

    # Define the normal distribution
    norm = lambda m, s, x: np.exp(-(x-m)**2/(2*s**2))/(s*(2*np.pi)**(1/2))

    # Set up bins for graphing normal distribution
    bins = np.arange(
        start = min(resids),
        stop = max(resids),
        step = (max(resids) - min(resids)) / 100
    )

    # First test plot
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("Is the data [L]inear?")
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.scatter(true, pred)
    ax.plot((min(true), max(true)),
            (min(pred), max(pred)), color="red")

    # Second test plot
    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("Is the error [I]ndependent along the x-axis?")
    ax.set_xlabel("Series")
    ax.set_ylabel("Residuals")
    ax.scatter(range(len(resids)), resids)
    ax.axhline(0, color="gray")

    # Third test plot
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("Is the error [N]ormal?")
    ax.set_xlabel("Residuals")
    ax.hist(resids, density=True)
    ax.plot(bins, norm(np.mean(resids), np.std(resids), bins), color="red")

    # Fourth test plot
    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("Is the variance of error [E]qual everywhere?")
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Residuals")
    ax.scatter(pred, resids)
    ax.axhline(0, color="gray")

    plt.tight_layout()
