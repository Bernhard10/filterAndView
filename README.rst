# filterAndView
Interactive analysis of data from the commandline

# Example

Example usage:

.. code:: python

   from fav import base, plotting, units
   class Analysis(units.UnitsMixin, plotting.PlotMixin, base.DataAnalysis): pass
   import pandas as pd
   df = pd.read_csv("my.csv")
   a=Analysis(df)
   a.show_help()
