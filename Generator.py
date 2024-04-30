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
        self.laplace_phi = np.zeros((4,4), dtype=complex)
        self.__update_phi(0)
        self.__laplace_phi(0)
        self.intensity = self.M.copy()
        self.meanintensity = solve(np.eye(4) - self.laplace_phi, self.M)
        self.meanintensity = np.real(self.meanintensity).reshape(4,)

        # Initialize the time of the point process
        self.T = T

        # Intialize the sequence of events for each point process (T-, T+, N-, N+)
        self.jumptimes = {0: np.empty((0,0), dtype=float), 1: np.empty((0,0), dtype=float), 2: np.empty((0,0), dtype=float), 3: np.empty((0,0), dtype=float)}
        self.alljumptimesN = np.empty((0,0), dtype=float)
        self.alljumptimesT = np.empty((0,0), dtype=float)

        # Initialize the counting process (T-, T+, N-, N+)
        self.countingprocess = {0: np.zeros((1,1), dtype=int), 1: np.zeros((1,1), dtype=int), 2: np.zeros((1,1), dtype=int), 3: np.zeros((1,1), dtype=int)}
        
        print(self.__laplace_priceautocovariance(1, 1))
        print(self.__laplace_tradeautocovariance(1, 1))
        pass

    def __phiTs(self, t: float) -> float:
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
        
    def __laplace_phiTs(self, z: float) -> complex:
        """
        Compute the laplace transformation of the phiTs function at time t

        Parameters:
        -----------
        - z: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - float
            The value of the function at time t
        """
        if type(z) == np.ndarray:
            return np.sum(0.6 / (20j * z + 1))
        else:
            return 0.6 / (20j * z + 1)

    def __phiNc(self, t: float) -> float:
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

    def __laplace_phiNc(self, z: float) -> complex:
        """
        Compute the laplace transformation of the phiNc function at time t

        Parameters:
        -----------
        - z: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - float
            The value of the function at time t
        """
        if type(z) == np.ndarray:
            return np.sum(0.5 / (10j * z + 1))
        else:
            return 0.5 / (10j * z + 1)
        
    def __phiIs(self, t:float) -> float:
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

    def __laplace_phiIs(self, z: float) -> complex:
        """
        Compute the laplace transformation of the phiIs function at time t

        Parameters:
        -----------
        - z: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - float
            The value of the function at time t
        """
        if type(z) == np.ndarray:
            return np.sum(25 / (1j * z + 100))
        else:
            return 25 / (1j * z + 100)
        
    def __phiFc(self, t: float) -> float:
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

    def __laplace_phiFc(self, z: float) -> complex:
        """
        Compute the laplace transformation of the phiFc function at time t

        Parameters:
        -----------
        - z: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - float
            The value of the function at time t
        """
        if type(z) == np.ndarray:
            return np.sum(0.2 / (2j * z + 1))
        else:
            return 0.2 / (2j * z + 1)
        
    def __update_phi(self, t: float) -> None:
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
    
    def __laplace_phi(self, z: float) -> None:
        """
        Calculate the laplace transformation of the phi matrix at time t

        Parameters:
        -----------
        - z: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - None
        """
        self.laplace_phi[0, 0] = self.__laplace_phiTs(z)
        self.laplace_phi[1, 1] = self.laplace_phi[0, 0]
        self.laplace_phi[2, 3] = self.__laplace_phiNc(z)
        self.laplace_phi[3, 2] = self.laplace_phi[2, 3]
        self.laplace_phi[2, 0] = self.__laplace_phiIs(z)
        self.laplace_phi[3, 1] = self.laplace_phi[2, 0]
        self.laplace_phi[0, 3] = self.__laplace_phiFc(z)
        self.laplace_phi[1, 2] = self.laplace_phi[0, 3]
        return None
    
    def __g(self, t: float, h: float) -> float:
        """
        Compute the g function at time t

        Parameters:
        -----------
        - t: float
            The time at which the function is evaluated
        - h: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - float
            The value of the function at time t
        """
        return np.maximum(0, 1 - np.abs(t) / h)
    
    def __laplace_g(self, z: float, h: float) -> complex:
        """
        Compute the laplace transformation of the g function at time h

        Parameters:
        -----------
        - z: float
            The value at which the function is evaluated
        - h: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - complex
            The value of the function at time h
        """
        return (-1j * h * z + - np.exp(-1j * h * z) + 1) / (h * z ** 2)

    def __laplace_delta_phiT(self, z: float) -> complex:
        """
        Compute the laplace transformation of delta of the phiTs function at time t

        Parameters:
        -----------
        - z: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - complex
            The value of the function at time t
        """
        return self.__laplace_phiTs(z)
    
    def __laplace_delta_phiN(self, z: float) -> complex:
        """
        Compute the laplace transformation of delta of the phiNc function at time t

        Parameters:
        -----------
        - z: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - complex
            The value of the function at time t
        """
        return - self.__laplace_phiNc(z)
    
    def __laplace_delta_phiI(self, z: float) -> complex:
        """
        Compute the laplace transformation of delta of the phiIs function at time t

        Parameters:
        -----------
        - z: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - complex
            The value of the function at time t
        """
        return self.__laplace_phiIs(z)
    
    def __laplace_delta_phiF(self, z: float) -> complex:
        """
        Compute the laplace transformation of delta of the phiFc function at time t

        Parameters:
        -----------
        - z: float
            The time at which the function is evaluated
        
        Returns:
        --------
        - complex
            The value of the function at time t
        """
        return - self.__laplace_phiFc(z)

    def __laplace_priceautocovariance(self, z: float, h: float) -> complex:
        num = 2 * self.__laplace_g(z, h) * (self.meanintensity[0] * np.linalg.norm(self.__laplace_delta_phiI(z)) ** 2 + self.meanintensity[2] * np.linalg.norm(1 - self.__laplace_delta_phiT(z)) ** 2)
        den = np.linalg.norm((1 - self.__laplace_delta_phiT(z)) * (1 - self.__laplace_delta_phiN(z)) - self.__laplace_delta_phiI(z) * self.__laplace_delta_phiF(z)) ** 2
        return num / den
    
    def __laplace_tradeautocovariance(self, z: float, h: float) -> complex:
        num = 2 * self.__laplace_g(z, h) * (self.meanintensity[0] * np.linalg.norm(1 - self.__laplace_delta_phiN(z)) ** 2 + self.meanintensity[2] * np.linalg.norm(self.__laplace_delta_phiF(z)) ** 2)
        den = np.linalg.norm((1 - self.__laplace_delta_phiT(z)) * (1 - self.__laplace_delta_phiN(z)) - self.__laplace_delta_phiI(z) * self.__laplace_delta_phiF(z)) ** 2
        return num / den
    

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
        
        self.__create_Xt()
        self.__create_Ut()

        return None
    
    def simulate_realtime(self, s: float,U, X) -> None:
        if s == 0:
            self.__reset()
        intensitymax = np.sum(self.intensity)
        if s < self.T:
            if s > 0:
                self.__update_intensities(s)
                intensitymax = np.sum(self.intensity)

            # Generate the time of the next event
            w = np.random.exponential(1 / intensitymax)
            s += w
            self.__update_intensities(s)

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
                    if k == 0:
                        U -= 1
                    else:
                        U += 1

                else:
                    self.alljumptimesN = np.append(self.alljumptimesN, s)
                    if k == 2:
                        X -= 1
                    else:
                        X += 1

        return s, U, X


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