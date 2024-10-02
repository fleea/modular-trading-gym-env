from abc import ABC, abstractmethod
from gymnasium import spaces
import numpy as np
from typing import TypeVar, Generic


EnvType = TypeVar("EnvType")


class BaseObservation(ABC, Generic[EnvType]):
    @abstractmethod
    def get_space(self) -> spaces.Space:
        """
        Returns the gymnasium space that describes the observation.
        """
        pass

    @abstractmethod
    def get_observation(self, env: EnvType) -> np.ndarray:
        """
        Returns the current observation based on the environment state.
        """
        pass

    def get_start_padding(self) -> int:
        """
        Returns the padding required for the observation.
        Default is 0 if not implemented in the subclass.
        Also means the current_step set in the initialisation of the environment state.
        """
        return 0
