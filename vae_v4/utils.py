from collections import UserList
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import numpy as np



class List_(UserList):

    def __init__(self, *args: Tuple, **kwargs: Dict) -> None:
        super().__init__(*args, **kwargs)


    def append(self, item) -> None:
        pred, act = item
        super().append(pred if pred == act else np.random.choice(item, 1))
