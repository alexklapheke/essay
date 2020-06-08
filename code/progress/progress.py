"""
File: progress.py
Author: Alex Klapheke
Email: alexklapheke@gmail.com
Github: https://github.com/alexklapheke
Description: Show progress meter for long iterations

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

from IPython.display import clear_output
from time import time


def _fmt_time(seconds):
    """Format an integer representing seconds as a time interval."""
    if seconds < 3600:
        return "{:.0f}:{:02.0f}".format(
                seconds // 60,  # M
                seconds % 60    # S
                )
    else:
        return "{:.0f}:{:02.0f}:{:02.0f}".format(
                (seconds // 3600),       # H
                (seconds % 3600) // 60,  # M
                seconds % 60             # S
                )


def _prog_bar(prog, width=30):
    """Show progress bar for progress `prog` between 0 and 1."""
    assert 0 <= prog <= 1, "Illegal value passed."
    bars = round(prog * width)
    return "[" + \
           "=" * bars + \
           ">" + \
           " " * (width - bars) + \
           "] " + \
           f"{int(prog * 100)}%"


def show_progress(function):
    """Shows progress of a loop run in an iPython environment, such as a
       Jupyter notebook. Say you are mapping a function `do_stuff_to` over
       a list called `my_list`. Define a function:

           @show_progress
           def my_fun(i, item):
               ... do_stuff_to(item) ...

       where `item` is a list item and `i` is its index. Then run:

           my_fun(my_list)

       You will see a running update of the progress of the function.
       You can modify the display by passing options to `my_fun`:

           update_freq: How many items are processed before the timer updates.
           progress_bar_width: The width of the progress bar in "equals signs".
                               Pass 0 to disable the progress bar altogether.
    """
    def progress_wrapper(iterable, update_freq=10, progress_bar_width=30):
        # Initialize
        it = list(iterable)
        n = len(it)
        start = time()
        for i, item in enumerate(it):

            # Compute remaining time
            current = time()
            if i % update_freq == 0:
                remaining = (current - start)/(i+1) * (n-(i+1))

            # Show progress
            clear_output(wait=True)
            prog = _prog_bar(i/n, width=progress_bar_width) \
                if progress_bar_width else f"({int(100*i/n)}%)"
            print("Parsing item {:,} of {:,} {}, ".format(i+1, n, prog) +
                  "{} remains...".format(_fmt_time(remaining)))

            # Run function
            function(i, item)

        # Compute total elapsed time
        elapsed = (time() - start)

        # Show completion
        clear_output(wait=True)
        print("Done! Parsed {:,} items in {}.".format(n, _fmt_time(elapsed)))

    return progress_wrapper
