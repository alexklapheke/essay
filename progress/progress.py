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
import time

def _format_time(seconds):
    """Format an integer representing seconds as a time interval."""
    if seconds < 3600:
        return "{:.0f}:{:02.0f}".format(
                (seconds % 3600) // 60, # M
                seconds % 60            # S
                )
    else:
        return "{:.0f}:{:02.0f}:{:02.0f}".format(
                (seconds // 3600),      # H
                (seconds % 3600) // 60, # M
                seconds % 60            # S
                )

def show_progress(function):
    """Shows progress of a loop run in an iPython environment, such as
       a Jupyter notebook. Say you are mapping a function my_fun over
       a list called my_list. Define a function:

           @show_progress
           def my_fun(i, item):
               ... do_stuff_to(item) ...

       where item is the item and i is its index in the list. Then run:

           my_fun(my_list)
    """
    def wrapper(iterable, update_freq=10):
        # Initialize
        l = list(iterable)
        n = len(l)
        start = time.time()
        for i, item in enumerate(l):

            # Compute remaining time
            current = time.time()
            if i % update_freq == 0:
                remaining = (current - start)/(i+1) * (n-(i+1))

            # Show progress
            clear_output(wait=True)
            print(f"Parsing item {i} of {n} ({int(100*i/n)}%), " +
                  f"{_format_time(remaining)} remains...")

            # Run function
            function(i, item)

        # Compute total elapsed time
        elapsed = (time.time() - start)

        # Show completion
        clear_output(wait=True)
        print(f"Done! Parsed {n} items in {_format_time(elapsed)}.")

    return wrapper
