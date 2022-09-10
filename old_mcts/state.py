import numpy as np
from dataclasses import dataclass

@dataclass
class State:
    observation: np.ndarray
    legal_moves: np.ndarray

    @property
    def state_id(self) -> str:
        # TODO figure out if we need to indicate whose turn it is, but I think not
        # This is faster than converting the np array to a string.
        return self.observation.tobytes()
