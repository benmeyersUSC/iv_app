import matplotlib.pyplot as plt
import numpy as np

def plot_diagram():


    # Add labels to the lines
    plt.text(.15, 0.84, 'Call (C)', verticalalignment='center', horizontalalignment='right', fontweight='bold')
    plt.text(0.23, 0.3, 'Portfolio to \nmimic Call (C)', verticalalignment='center', horizontalalignment='right',
             fontweight='bold')

    plt.text(.27, .62, 'no cashflows', rotation=-27)
    plt.text(.26, .6, '--------------------------->', rotation=-27)

    plt.text(.26, .34, 'no (NET) cashflows', rotation=27)
    plt.text(.26, .3, '--------------------------->', rotation=27)

    plt.text(.58, .535, '-------------------------------------')

    plt.text(.72, .88, 'S > K:', fontweight='bold')
    plt.text(.67, .7, 'C = S - K', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, pad=10), color='g')
    plt.text(.72, .45, 'S < K:', fontweight='bold')
    plt.text(.7, .3, 'C = 0', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, pad=10), color='r')


    # plt.text(.05, .12, 'PV(C) if S > K = (S - K) * e^-(rt)...EV(PV(C)) = DELTA * (S - K) * e^-(rt)', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, pad=10))
    plt.text(.09, .1, r'PV(C) if $S > K$ = $\mathbf{(S - K) * e^{-rt}}$...EV(PV(C)) ~=~ $\mathbf{\Delta}$ * '
                       r'$\mathbf{(S - K) * e^{-rt}}$', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, pad=10))


    plt.xticks([])
    plt.yticks([])
    # Show plot
    plt.savefig('arbitrage_payoffs.png')
    plt.show()

# Call the function to generate and display the plot
plot_diagram()