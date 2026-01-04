
from envs.hanoi_env import HanoiEnv
from copy import deepcopy

def run_full_hanoi_suite(classe, n_disks, debug=False):
    env = classe(n_disks)
    
    state = env.reset()
    assert state["A"] == list(range(n_disks, 0, -1))
    assert state["B"] == []
    assert state["C"] == []
    
    if n_disks >= 2:
        env.reset()
        env.step(('A', 'B'))
        state, ok, done, info = env.step(('A', 'B'))
        assert ok is False
        assert info == "ILLEGAL_MOVE_LARGER_ON_SMALLER"
        assert env.pegs['B'] == [1]

    env.reset()
    state, ok, done, info = env.step(('A', 'B'))
    all_disks = env.pegs['A'] + env.pegs['B'] + env.pegs['C']
    assert len(all_disks) == len(set(all_disks))
    assert len(all_disks) == n_disks

    env.reset()
    env.step(('A', 'B'))
    state = env.get_state()
    assert state['A'][-1] == n_disks - 1 if n_disks > 1 else True
    assert state['B'][-1] == 1

    env.reset()
    initial_state = deepcopy(env.get_state())
    state, ok, done, info = env.step(('B', 'C'))
    assert ok is False
    assert state == initial_state
    assert env.step_counter == 0

    env.reset()
    env.step(('A', 'B'))
    assert env.step_counter == 1
    env.step(('B', 'C'))
    assert env.step_counter == 2
    env.step(('B', 'A'))
    assert env.step_counter == 2
    assert len(env.history) == 2

    small_env = classe(1)
    small_env.step(('A', 'C'))
    assert small_env.done is True
    try:
        small_env.step(('C', 'B'))
    except RuntimeError:
        pass
    assert small_env.done is True

    env.reset()
    actions = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    for act in actions:
        env.step(act)
    state_one = deepcopy(env.get_state())
    
    env.reset()
    for act in actions:
        env.step(act)
    state_two = deepcopy(env.get_state())
    assert state_one == state_two

    if debug:
        print("All 8 environment tests passed successfully.")

run_full_hanoi_suite(HanoiEnv, 4, debug=True)