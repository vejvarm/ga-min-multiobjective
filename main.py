from matplotlib import pyplot as plt

import ga
from structures import Problem, Params, f1, f2

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_title("Objective space and Pareto front (hopefully)")
    ax.set_xlim(0, 10)
    ax.set_ylim(-10, 2)

    # Run GA
    epochs = 10
    for e in range(epochs):
        out = ga.run(Problem, Params)

    # calculate objective values
        for p in out.pop:
            y1 = f1(p.position)
            y2 = f2(p.position)

            # Results
            plt.stem(y1, y2, linefmt="none", markerfmt="rx", basefmt="")

    plt.show()

# constraints (nonlinear, sadly :/)
# h1:   x1    + 2*x2    -     x3    - 0.5*x4    +     x5    -  2.0 == 0.0
# h2: 4*x1    - 2*x2    + 0.8*x3    + 0.6*x4    + 0.5*x5**2 +  0.0 == 0.0
# g1:   x1**2 +   x2**2 +     x3**2 +     x4**2 +     x5**2 - 10.0 <= 0.0