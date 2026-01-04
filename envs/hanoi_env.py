from envs.base_env import BaseEnv
from copy import deepcopy

class HanoiEnv(BaseEnv):
    def __init__(self, n_disks: int):
        """here we initialise step counter done flag and n disks"""
        assert isinstance(n_disks, int), "n_disks must be an integer"
        assert n_disks >= 1, "n_disks must be >= 1"
        self.disks = n_disks
        self.reset()

    def reset(self):
        """start a fresh here re initialise pegs like store on a then make done= False and step counter to 0 """
        self.pegs = {
            "A": list(range(self.disks, 0, -1)),
            "B": [],
            "C": []
        }
        self.step_counter = 0
        self.done = False
        self.history = []
        return self.get_state()

    def get_state(self):
        return deepcopy(self.pegs)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError("Action must be a tuple (src, dest)")

        src, dest = action

        if src not in self.pegs or dest not in self.pegs:
            raise ValueError("Invalid peg name")

        ok = True
        info = None

        if not self.pegs[src]:
            ok = False
            info = "EMPTY_SOURCE"
        else:
            disk = self.pegs[src][-1]
            if self.pegs[dest] and self.pegs[dest][-1] < disk:
                ok = False
                info = "ILLEGAL_MOVE_LARGER_ON_SMALLER"

        if ok:
            self.pegs[src].pop()
            self.pegs[dest].append(disk)
            self.step_counter += 1
            self.history.append(action)

            if self.pegs["C"] == list(range(self.disks, 0, -1)):
                self.done = True

        return self.get_state(), ok, self.done, info
