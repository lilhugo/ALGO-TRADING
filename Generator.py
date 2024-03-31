from Packages import *

class PointProcess:

    def __init__(self, mu: float, T: float) -> None:
        """
        Initialize the Point Process

        Parameters:
        -----------
        - mu: float 
            The initial intensity of the point process
        - T: float
            The time of the point process

        Returns:
        --------
        - None
        """
    
        # Initialize the intensity matrix
        self.mu = mu
        self.M = np.zeros((4,1))
        self.M[:2] = self.mu
        self.phi = np.zeros((4,4))
        self.__update_phi(0)
        self.intensity = self.M.copy()

        # Initialize the time of the point process
        self.T = T

        # Intialize the sequence of events for each point process (T-, T+, N-, N+)
        self.jumptimes = {0: np.empty((0,0), dtype=float), 1: np.empty((0,0), dtype=float), 2: np.empty((0,0), dtype=float), 3: np.empty((0,0), dtype=float)}
        self.alljumptimesN = np.empty((0,0), dtype=float)
        self.alljumptimesT = np.empty((0,0), dtype=float)

        # Initialize the counting process (T-, T+, N-, N+)
        self.countingprocess = {0: np.zeros((1,1), dtype=int), 1: np.zeros((1,1), dtype=int), 2: np.zeros((1,1), dtype=int), 3: np.zeros((1,1), dtype=int)}
        
        return None

    def __phiTs(self, t) -> float:
        """
        Compute the phiTs function at time t

        Parameters:
        -----------
        - t: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - float
            The value of the function at time t
        """
        if type(t) == np.ndarray:
            return np.sum(0.03 * np.exp(-5 * t / 100))
        else:
            return 0.03 * np.exp(-5 * t / 100)

    def __phiNc(self, t) -> float:
        """
        Compute the phiNc function at time t

        Parameters:
        -----------
        - t: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - float
            The value of the function at time t
        """
        if type(t) == np.ndarray:
            return np.sum(0.05 * np.exp(-t / 10))
        else:
            return 0.05 * np.exp(-t / 10)

    def __phiIs(self, t) -> float:
        """
        Compute the phiIs function at time t

        Parameters:
        -----------
        - t: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - float
            The value of the function at time t
        """
        if type(t) == np.ndarray:
            return np.sum(25 * np.exp(-100 * t))
        else:
            return 25 * np.exp(-100 * t)     

    def __phiFc(self, t) -> float:
        """
        Compute the phiFc function at time t

        Parameters:
        -----------
        - t: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - float
            The value of the function at time t
        """
        if type(t) == np.ndarray:
            return np.sum(0.1 * np.exp(-t / 2))
        else:
            return 0.1 * np.exp(-t / 2)

    def __update_phi(self, t) -> None:
        """
        Update the phi matrix at time t

        Parameters:
        -----------
        - t: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - None
        """
        self.phi[0, 0] = self.__phiTs(t)
        self.phi[1, 1] = self.phi[0,0]
        self.phi[2, 3] = self.__phiNc(t)
        self.phi[3, 2] = self.phi[2, 3]
        self.phi[2, 0] = self.__phiIs(t)
        self.phi[3, 1] = self.phi[2, 0]
        self.phi[0, 3] = self.__phiFc(t)
        self.phi[1, 2] = self.phi[0, 3]
        return None

    def __update_intensities(self, t: float) -> None:
        """
        Update the intensity matrix at time t

        Parameters:
        -----------
        - t: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - None
        """
        self.intensity = self.M.copy()
        for i in range(4):
            basisvector = np.zeros((4,1))
            basisvector[i] = 1
            if self.jumptimes[i].size != 0:
                self.__update_phi(t - self.jumptimes[i])
                self.intensity += self.phi @ basisvector
        return None
    
    def __reset(self) -> None:
        """
        reset the point process

        Parameters:
        -----------
        - None

        Returns:
        --------
        - None
        """
        # Initialize the intensity matrix
        self.intensity = self.M.copy()

        # Intialize the sequence of events for each point process (T-, T+, N-, N+)
        self.jumptimes = {0: np.empty((0,0), dtype=float), 1: np.empty((0,0), dtype=float), 2: np.empty((0,0), dtype=float), 3: np.empty((0,0), dtype=float)}
        self.alljumptimesN = np.empty((0,0), dtype=float)
        self.alljumptimesT = np.empty((0,0), dtype=float)

        # Initialize the counting process (T-, T+, N-, N+)
        self.countingprocess = {0: np.zeros((1,1), dtype=int), 1: np.zeros((1,1), dtype=int), 2: np.zeros((1,1), dtype=int), 3: np.zeros((1,1), dtype=int)}

        return None


    def __create_Ut(self) -> None:
        """
        Create the difference of the counting process of the T+ and T- processes by interpolate between the different jumps

        Parameters:
        -----------
        - None

        Returns:
        --------
        - None
        """
        interpT_minus = interp1d(self.jumptimes[0],self.countingprocess[0], kind='nearest', fill_value="extrapolate")
        T_minus = interpT_minus(self.alljumptimesT)
        interpT_plus = interp1d(self.jumptimes[1],self.countingprocess[1], kind='nearest', fill_value="extrapolate")
        T_plus = interpT_plus(self.alljumptimesT)
        self.Ut = T_plus - T_minus
        return None

    def __create_Xt(self) -> None:
        """
        Create the difference of the counting process of the N+ and N- processes by interpolate between the different jumps

        Parameters:
        -----------
        - None

        Returns:
        --------
        - None
        """
        interpN_minus = interp1d(self.jumptimes[2],self.countingprocess[2], kind='nearest', fill_value="extrapolate")
        N_minus = interpN_minus(self.alljumptimesN)
        interpN_plus = interp1d(self.jumptimes[3],self.countingprocess[3], kind='nearest', fill_value="extrapolate")
        N_plus = interpN_plus(self.alljumptimesN)
        self.Xt = N_plus - N_minus
        return None


    def simulate(self) -> None:
        """
        Simulate the point process until time T

        Parameters:
        -----------
        - None

        Returns:
        --------
        - None
        """   
        self.__reset()
        s = 0 
        intensitymax = np.sum(self.intensity)
        lastjump = None
        while s < self.T:
            if s > 0:
                self.__update_intensities(s)
                intensitymax = np.sum(self.intensity)

            # Generate the time of the next event
            w = np.random.exponential(1 / intensitymax)
            s += w
            self.__update_intensities(s)

            if s > self.T:
                break

            # Generate the type of the event
            D = np.random.uniform(0, 1)
            if D <= np.sum(self.intensity) / intensitymax:
                k = 0
                while D * intensitymax > np.sum(self.intensity[:k+1]):
                    k += 1
                if self.countingprocess[k][0] == 0:
                    self.countingprocess[k][0] = 1
                    self.jumptimes[k] = np.append(self.jumptimes[k], s)
                else:
                    self.countingprocess[k] = np.append(self.countingprocess[k], self.countingprocess[k][-1] + 1)
                    self.jumptimes[k] = np.append(self.jumptimes[k], s)
                if k == 0 or k == 1:
                    self.alljumptimesT = np.append(self.alljumptimesT, s)
                else:
                    self.alljumptimesN = np.append(self.alljumptimesN, s)
                lastjump = k
        
        self.__create_Xt()
        self.__create_Ut()

        return None

    def plot(self, plot: str="both") -> None:
        """
        Plot the counting process. This can plot the counting process of the T process, the N process or both

        Parameters:
        -----------
        - plot: str
            The type of the counting process to plot. It can be "T", "N" or "both"
        
        Returns:
        --------
        - None
        """
        if plot not in ["T", "N", "both"]:
            raise ValueError("The plot argument must be 'T', 'N' or 'both'")
        
        if plot != "both":
            plt.figure(figsize=(12, 4))
            plt.grid()
            if plot == "N":
                plt.step(self.alljumptimesN, self.Xt,c="black", linewidth=.5)
                plt.ylabel('Xt')
            else:
                plt.step(self.alljumptimesT, self.Ut,c="black", linewidth=.5)
                plt.ylabel('Ut')
            plt.xlabel('Time (s)')

        else:
            fig, ax = plt.subplots(2, 1, figsize=(12, 8))
            ax[0].step(self.alljumptimesT, self.Ut,c="black", linewidth=.5)
            ax[0].set_ylabel('Ut')
            ax[0].set_xlabel('Time (s)')
            ax[0].grid()
            ax[1].step(self.alljumptimesN, self.Xt,c="black", linewidth=.5)
            ax[1].set_ylabel('Xt')
            ax[1].set_xlabel('Time (s)')
            ax[1].grid()

        plt.show()
        return None