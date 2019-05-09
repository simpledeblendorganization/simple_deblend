# Some temporary code to analyze how close a period should be to be regarded
# as the same


import numpy as np
import matplotlib.pyplot as plt

df = 1./78.

periods = np.linspace(0.06,75.,10000)

frequencies = 1./periods

deltaP = 1./(frequencies - df) - 1./(frequencies+df)

plt.plot(periods,deltaP)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("period")
plt.ylabel("delta period")
plt.savefig('temp.pdf')
