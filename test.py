import numpy as np
import sophus as sp

b = np.array([-0.000700019, 0.00482854, -0.00345816, -0.000251848, 0.00178883, -0.042933])

print(sp.SE3.exp(b).inverse())