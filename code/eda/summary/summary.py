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
    dfg = df.groupby(col)
    return pd.concat([
            dfg.first().T.add_prefix("head_"),
            dfg.dtypes.T.add_prefix("type_"),
            dfg.apply(lambda x: x.isna().sum()).T.add_prefix("nulls_"),
        ], axis=1)
