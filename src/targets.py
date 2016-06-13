import numpy as np
import abc


#Requirement: larger value must mean disruption more likely.
class Target(object):
    @abc.abstractmethod
    def remapper(self,ttd,T_warning):
        return -ttd

    @abc.abstractmethod
    def threshold_range():
        return np.logspace(-1,4,100)


class BinaryTarget(Target):
    def remapper(ttd,T_warning,as_array_of_shots=True):
        binary_ttd = 0*ttd
        mask = ttd < np.log10(T_warning)
        binary_ttd[mask] = 1.0
        binary_ttd[~mask] = 0.0
        return binary_ttd

    def threshold_range():
        return np.logspace(-6,0,100)


class TTDTarget(Target):
    def remapper(ttd,T_warning):
        mask = ttd < np.log10(T_warning)
        ttd[~mask] = np.log10(T_warning)
        return -ttd

    def threshold_range():
        return np.linspace(-np.log10(T_warning),6,100)



class TTDLinearTarget(Target):
    def remapper(ttd,T_warning):
        ttd = 10**(ttd)
        mask = ttd < T_warning
        ttd[~mask] = 0#T_warning
        ttd[mask] = T_warning - ttd[mask]#T_warning
        return ttd

    def threshold_range():
        return np.logspace(-6,log10(T_warning),100)


class HingeTarget(Target):
    def remapper(ttd,T_warning,as_array_of_shots=True):
        binary_ttd = 0*ttd
        mask = ttd < np.log10(T_warning)
        binary_ttd[mask] = 1.0
        binary_ttd[~mask] = -1.0
        return binary_ttd

    def threshold_range():
        return linspace(-2,2,100)

