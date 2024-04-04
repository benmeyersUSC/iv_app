import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def likelihood_from_std_dev(std_dev):

    return norm.pdf(std_dev)


x = np.arange(-3.6, 3.6, .01)
y = [likelihood_from_std_dev(n) for n in x]

plt.plot(x, y)
d1 = -.85
shade_x_g = np.arange(-3.6, d1, .01)
shade_x_r = np.arange(d1, 3.6, .01)
shade_y_g = [likelihood_from_std_dev(n) for n in shade_x_g]
shade_y_r = [likelihood_from_std_dev(n) for n in shade_x_r]

plt.fill_between(shade_x_g, shade_y_g, color='green', alpha=0.3)  # Alpha controls transparency
plt.fill_between(shade_x_r, shade_y_r, color='red', alpha=0.3)  # Alpha controls transparency

b = norm.cdf(d1)
plt.text(.8, 0.35, f'b = {b:.4f}', fontsize=12)


plt.axvline(x=d1, ymin=.05, ymax=.67, color='red', linestyle='-')
plt.axvline(x=0, ymin=.05, ymax=.95, color='gray', linestyle='-')
plt.text(d1, 0.1, 'd2 = -0.85', color='red', fontsize=12, ha='center')


plt.xlabel('<---d2--->')
plt.ylabel('<---Probability--->')

plt.savefig('b_curve_2.png')

plt.show()