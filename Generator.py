from Packages import *

class PointProcess:

    def __init__(self, mu: float) -> None:
        self.mu = mu
        self.PointProcess = np.zeros((4,4))
        self.intensity = np.zeros((4,4))
        self.M = np.zeros((4,4))
        self.M[:2] = self.mu
        self.phi = np.zeros((4,4))
        self.update_phi(0)
        return None

    def phiTs(self, t) -> float:
        return 0.03 * np.exp(-5 * t / 100)

    def phiNc(self, t) -> float:
        return 0.05 * np.exp(-t / 10)

    def phiIs(self, t) -> float:
        return 25 * np.exp(-100 * t)

    def phiFc(self, t) -> float:
        return 0.1 * np.exp(-t / 2)

    def update_phi(self, t) -> None:
        self.phi[0, 0] = self.phiTs(t)
        self.phi[1, 1] = self.phi[0,0]
        self.phi[2, 3] = self.phiNc(t)
        self.phi[3, 2] = self.phi[2, 3]
        self.phi[2, 0] = self.phiIs(t)
        self.phi[3, 1] = self.phi[2, 0]
        self.phi[0, 3] = self.phiFc(t)
        self.phi[1, 2] = self.phi[0, 3]
        return None