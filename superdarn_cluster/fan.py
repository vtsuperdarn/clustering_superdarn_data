import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

class FanPlot:

    def __init__(self, nrange=75, nbeam=16, rinit=180, dr=45, dtheta=3.3):
        # Set member variables
        self.nrange = nrange
        self.nbeam = nbeam
        self.rinit = 180
        self.dr = dr
        self.dtheta = dtheta
        # Initial angle (from X, polar coordinates) for beam 0
        self.theta0 = 90 - dtheta * nbeam / 2

        # Create axis
        fig = plt.figure()
        self.ax = fig.add_subplot(111, polar=True)

        # Set up min/max and ticks
        r_ticks = range(rinit, rinit + nrange * dr, dr)
        theta_ticks = np.linspace(self.theta0, self.theta0 + nbeam * dtheta, nbeam)
        ax.set_thetamin(theta_ticks[0])
        ax.set_thetamax(theta_ticks[-1])
        ax.set_rmin(0)
        ax.set_rmax(r_ticks[-1])
        plt.rgrids(r_ticks, range(nrange))
        plt.thetagrids(theta_ticks, range(nbeam))

    def plot(self, beams, gates, color):
        self.ax.scatter(beams, gates)


"""
nbeam = 16
nrange = 75
rinit = 180
dr = 45
dtheta = 3.3

# Initial angle (from X, polar coordinates) for beam 0
theta0 = 90 - dtheta * nbeam / 2

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
r_ticks = range(rinit, rinit + nrange * dr, dr)
theta_ticks = np.linspace(self.theta0, self.theta0 + nbeam * dtheta, nbeam)

# Compute areas and colors
r = range(rinit, rinit + nrange * dr, dr)
theta = np.linspace(theta0, theta0 + nbeam * dtheta, len(r)) # range(theta0, theta0 + nbeam * dtheta, dtheta)
colors = theta


bottom_right = (3.14 / 2, 1)
ax.add_patch(
    patches.Rectangle(
        bottom_right, width=3.14 * 0.25, height=0.5
    )
)
# hacky bugfix - awaiting https://github.com/matplotlib/matplotlib/issues/8521
ax.bar(0,1).remove()

r_ticks = r
theta_ticks = np.linspace(theta0, theta0 + nbeam * dtheta, nbeam)

#ax.set_thetamin(theta_ticks[0])
#ax.set_thetamax(theta_ticks[-1])
#ax.set_rmin(0)
#ax.set_rmax(r_ticks[-1])

#plt.rgrids(r_ticks, range(nrange))
#plt.thetagrids(theta_ticks, range(nbeam))


plt.show()

"""