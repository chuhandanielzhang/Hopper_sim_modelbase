


import numpy as np
from hopper_config import HopperConfig
from simulink_velocity_filter import SimulinkVelocityFilterVector

class ForwardKinematics:
    
    def __init__(self, config=None):
        if config is None:
            config = HopperConfig()
        self.D = config.D
        self.d = config.d
        self.r = config.r
        
    def forward_kinematics(self, theta):
        s1 = np.sin(theta[0])
        s2 = np.sin(theta[1])
        s3 = np.sin(theta[2])
        c1 = np.cos(theta[0])
        c2 = np.cos(theta[1])
        c3 = np.cos(theta[2])
        
        pos = np.zeros(3)
        D, d, r = self.D, self.d, self.r
        

        pos[0] = (3 ** (1 / 2) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3)) - (3 ** (1 / 2) * r) / 2 - (3 ** (1 / 2) * D * c2) / 2 + (3 ** (1 / 2) * D * (((2 * D * s2 + (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 - (D * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2)) ** 2 - ((4 * ((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) ** 2) / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 + (D ** 2 * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) ** 2) / (3 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) + 4) * ((3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) ** 2 / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) - D ** 2 * (c2 ** 2 - 1) - d ** 2 + (3 * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) ** 2) / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2))) ** (1 / 2) / 2 + D * s2 + (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) - (D * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2)) * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2]))) / (6 * (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) ** 2 / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 + (D ** 2 * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) ** 2) / (12 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) + 1) * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3))
        

        pos[1] = (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3)) - r / 2 - (D * c2) / 2 - (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (((2 * D * s2 + (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 - (D * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2)) ** 2 - ((4 * ((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) ** 2) / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 + (D ** 2 * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) ** 2) / (3 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) + 4) * ((3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) ** 2 / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) - D ** 2 * (c2 ** 2 - 1) - d ** 2 + (3 * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) ** 2) / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2))) ** (1 / 2) / 2 + D * s2 + (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) - (D * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2))) / ((((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) ** 2 / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 + (D ** 2 * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) ** 2) / (12 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) + 1) * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3))
        


        pos[2] = (((2 * D * s2 + (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 - (D * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2)) ** 2 - ((4 * ((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) ** 2) / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 + (D ** 2 * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) ** 2) / (3 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) + 4) * ((3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) ** 2 / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) - D ** 2 * (c2 ** 2 - 1) - d ** 2 + (3 * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) ** 2) / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2))) ** (1 / 2) / 2 + D * s2 + (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) - (D * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2)) / (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) ** 2 / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 + (D ** 2 * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) ** 2) / (12 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) + 1)
        pos[0] = pos[0]
        pos[2] = pos[2]
        pos[1] = pos[1]





        knee_positions = np.zeros((3, 3))
        for i in range(3):
            angle_deg = 120 * i
            R_i = np.array([[np.cos(np.deg2rad(angle_deg)), -np.sin(np.deg2rad(angle_deg)), 0],
                           [np.sin(np.deg2rad(angle_deg)), np.cos(np.deg2rad(angle_deg)), 0],
                           [0, 0, 1]])

            knee_base = np.array([0, self.r, 0])
            knee_offset = np.array([0, self.D*np.cos(theta[i]), self.D*np.sin(theta[i])])
            knee_positions[:, i] = R_i @ (knee_base + knee_offset)

        check = np.zeros(3)
        for i in range(3):
            check[i] = np.linalg.norm(pos - knee_positions[:, i]) - self.d
        
        return pos, check

class InverseJacobian:
    
    def __init__(self, config=None, use_simulink_filter=True, forgetting_factor=0.95, dt=0.001):
        if config is None:
            config = HopperConfig()
        self.D = config.D
        self.d = config.d
        self.r = config.r
        

        self.forward_kinematics_obj = ForwardKinematics(config)
        

        self.use_simulink_filter = use_simulink_filter
        if use_simulink_filter:
            self.simulink_filter = SimulinkVelocityFilterVector(dt=dt, forgetting_factor=forgetting_factor)
        else:
            self.simulink_filter = None
    
    def forward_kinematics(self, theta):
        return self.forward_kinematics_obj.forward_kinematics(theta)
        
    def inverse_jacobian(self, x, thetadot, theta=None):

        if self.use_simulink_filter and theta is not None:
            thetadot_filtered = self.simulink_filter.update(theta)
        else:
            thetadot_filtered = thetadot
        

        D, d, r = self.D, self.d, self.r
        

        j = np.zeros((3, 3))
        
        sqrt3 = np.sqrt(3)
        


        sqrt_arg1 = 1 - ((r - x[1])**2/2 + D**2/2 - d**2/2 + x[0]**2/2 + x[2]**2/2)**2/(D**2*((r - x[1])**2 + x[2]**2))
        sqrt_arg1 = max(sqrt_arg1, 1e-12)
        denom1 = D * np.sqrt((r - x[1])**2 + x[2]**2) * np.sqrt(sqrt_arg1)
        j[0, 0] = x[0] / denom1
        

        term1 = (r - x[1])/(D*np.sqrt((r - x[1])**2 + x[2]**2))
        term2 = ((2*r - 2*x[1])*((r - x[1])**2/2 + D**2/2 - d**2/2 + x[0]**2/2 + x[2]**2/2))/(2*D*((r - x[1])**2 + x[2]**2)**(3/2))
        sqrt_arg2 = 1 - ((r - x[1])**2/2 + D**2/2 - d**2/2 + x[0]**2/2 + x[2]**2/2)**2/(D**2*((r - x[1])**2 + x[2]**2))
        sqrt_arg2 = max(sqrt_arg2, 1e-12)
        denom2 = np.sqrt(sqrt_arg2)
        j[0, 1] = -(term1 - term2)/denom2 - x[2]/((r - x[1])**2 + x[2]**2)
        

        term3 = x[2]/(D*np.sqrt((r - x[1])**2 + x[2]**2))
        term4 = (x[2]*((r - x[1])**2/2 + D**2/2 - d**2/2 + x[0]**2/2 + x[2]**2/2))/(D*((r - x[1])**2 + x[2]**2)**(3/2))
        j[0, 2] = (term3 - term4)/denom2 - (r - x[1])/((r - x[1])**2 + x[2]**2)
        


        temp_x1 = x[0]/4 - sqrt3*x[1]/4
        temp_x2 = r + x[1]/2 + sqrt3*x[0]/2
        temp_x3 = x[0]/2 - sqrt3*x[1]/2
        temp_denom = np.sqrt((temp_x2)**2 + x[2]**2)
        temp_arg = (temp_x2**2/2 + D**2/2 - d**2/2 + temp_x3**2/2 + x[2]**2/2)
        sqrt_arg_temp = 1 - temp_arg**2/(D**2*((temp_x2)**2 + x[2]**2))
        sqrt_arg_temp = max(sqrt_arg_temp, 1e-12)
        temp_sqrt_denom = np.sqrt(sqrt_arg_temp)
        

        numerator1 = temp_x1 + sqrt3*temp_x2/2
        term1_1 = numerator1 / (D * temp_denom)
        term2_1 = (sqrt3 * temp_x2 * temp_arg) / (2 * D * ((temp_x2)**2 + x[2]**2)**(3/2))
        j[1, 0] = (term1_1 - term2_1) / temp_sqrt_denom + sqrt3*x[2] / (2*((temp_x2)**2 + x[2]**2))
        

        numerator2 = r/2 + x[1]/4 - sqrt3*temp_x3/2 + sqrt3*x[0]/4
        term1_2 = numerator2 / (D * temp_denom)
        term2_2 = (temp_x2 * temp_arg) / (2 * D * ((temp_x2)**2 + x[2]**2)**(3/2))
        j[1, 1] = x[2] / (2*((temp_x2)**2 + x[2]**2)) + (term1_2 - term2_2) / temp_sqrt_denom
        

        temp_big_expr = -2*D**2 + 2*d**2 + 2*r**2 + 2*sqrt3*r*x[0] + 2*r*x[1] + x[0]**2 + 2*sqrt3*x[0]*x[1] - x[1]**2 + 2*x[2]**2
        temp_denom_big = r*x[1] + D**2 - d**2 + r**2 + x[0]**2 + x[1]**2 + x[2]**2 + sqrt3*r*x[0]
        temp_final_denom = (2*r + x[1] + sqrt3*x[0])**2 + 4*x[2]**2
        sqrt_arg_final = 1 - temp_denom_big**2/(D**2 * temp_final_denom)
        sqrt_arg_final = max(sqrt_arg_final, 1e-12)
        temp_sqrt_final = np.sqrt(sqrt_arg_final)
        
        j[1, 2] = (2*x[2]*temp_big_expr) / (D * temp_sqrt_final * temp_final_denom**(3/2)) - (4*temp_x2) / temp_final_denom
        


        temp2_x1 = x[0]/4 + sqrt3*x[1]/4
        temp2_x2 = r + x[1]/2 - sqrt3*x[0]/2
        temp2_x3 = x[0]/2 + sqrt3*x[1]/2
        temp2_denom = np.sqrt((temp2_x2)**2 + x[2]**2)
        temp2_arg = (temp2_x2**2/2 + D**2/2 - d**2/2 + temp2_x3**2/2 + x[2]**2/2)
        sqrt_arg_temp2 = 1 - temp2_arg**2/(D**2*((temp2_x2)**2 + x[2]**2))
        sqrt_arg_temp2 = max(sqrt_arg_temp2, 1e-12)
        temp2_sqrt_denom = np.sqrt(sqrt_arg_temp2)
        

        numerator3 = temp2_x1 - sqrt3*temp2_x2/2
        term1_3 = numerator3 / (D * temp2_denom)
        term2_3 = (sqrt3 * temp2_x2 * temp2_arg) / (2 * D * ((temp2_x2)**2 + x[2]**2)**(3/2))
        j[2, 0] = (term1_3 + term2_3) / temp2_sqrt_denom - sqrt3*x[2] / (2*((temp2_x2)**2 + x[2]**2))
        

        numerator4 = r/2 + x[1]/4 + sqrt3*temp2_x3/2 - sqrt3*x[0]/4
        term1_4 = numerator4 / (D * temp2_denom)
        term2_4 = (temp2_x2 * temp2_arg) / (2 * D * ((temp2_x2)**2 + x[2]**2)**(3/2))
        j[2, 1] = x[2] / (2*((temp2_x2)**2 + x[2]**2)) + (term1_4 - term2_4) / temp2_sqrt_denom
        

        temp2_big_expr = -2*D**2 + 2*d**2 + 2*r**2 - 2*sqrt3*r*x[0] + 2*r*x[1] + x[0]**2 - 2*sqrt3*x[0]*x[1] - x[1]**2 + 2*x[2]**2
        temp2_denom_big = r*x[1] + D**2 - d**2 + r**2 + x[0]**2 + x[1]**2 + x[2]**2 - sqrt3*r*x[0]
        temp2_final_denom = (2*r + x[1] - sqrt3*x[0])**2 + 4*x[2]**2
        sqrt_arg_final2 = 1 - temp2_denom_big**2/(D**2 * temp2_final_denom)
        sqrt_arg_final2 = max(sqrt_arg_final2, 1e-12)
        temp2_sqrt_final = np.sqrt(sqrt_arg_final2)
        
        j[2, 2] = (2*x[2]*temp2_big_expr) / (D * temp2_sqrt_final * temp2_final_denom**(3/2)) - (4*temp2_x2) / temp2_final_denom
        


        try:
            xdot = np.linalg.solve(j, thetadot_filtered[:3])
        except np.linalg.LinAlgError:
            xdot = np.linalg.lstsq(j, thetadot_filtered[:3], rcond=None)[0]
        
        return j, xdot
