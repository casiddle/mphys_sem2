import numpy as np
import pandas as pd

points = np.linspace(0,10,100)
array_to_save = []
for point in points:
    rounded_point = round(point,1)
    rounded_point_1 = round(rounded_point+0.01, 2)
    rounded_point_2 = round(rounded_point-0.01, 2)
    array_to_save.append([rounded_point_1, rounded_point_2, rounded_point])

df = pd.DataFrame(array_to_save, columns=["CB Point", "Window length", "Unrounded Point"])
df.to_csv(r"Simulations\ContinueBackComparison\CB_point_array.csv", index=False)

