
from envs.hanoi_env import HanoiEnv
def hanoi_test(classe,n_disks):
    env=classe(n_disks)
    state=env.reset()
    assert isinstance(state,dict), "Rest did not give pegs snapshot"
    "for cheking ordering of disk"
    # env.step()
