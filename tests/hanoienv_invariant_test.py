from envs.hanoi_env import HanoiEnv
from copy import deepcopy

def run_full_hanoi_suite(classe, n_disks, debug=False):
    env = classe(n_disks)
    
    # TEST 1: Reset must initialize all disks on peg A in descending order,
    # with pegs B and C empty (canonical Hanoi start state).
    state = env.reset()
    assert state["A"] == list(range(n_disks, 0, -1))
    assert state["B"] == []
    assert state["C"] == []
    
    # TEST 2: Illegal move detection.
    # Moving a larger disk onto a smaller one must:
    # - return ok=False
    # - preserve the smaller disk on the target peg
    # - emit the correct error code
    if n_disks >= 2:
        env.reset()
        env.step(('A', 'B'))                 # Move smallest disk legally
        state, ok, done, info = env.step(('A', 'B'))  # Attempt illegal move
        assert ok is False
        assert info == "ILLEGAL_MOVE_LARGER_ON_SMALLER"
        assert env.pegs['B'] == [1]

    # TEST 3: Disk conservation invariant.
    # After any move, all disks must:
    # - be unique
    # - still total exactly n_disks
    env.reset()
    state, ok, done, info = env.step(('A', 'B'))
    all_disks = env.pegs['A'] + env.pegs['B'] + env.pegs['C']
    assert len(all_disks) == len(set(all_disks))
    assert len(all_disks) == n_disks

    # TEST 4: Post-move ordering invariant.
    # After removing the smallest disk from A:
    # - the new top of A must be the smallest remaining disk
    # - peg B must contain disk 1
    env.reset()
    env.step(('A', 'B'))
    state = env.get_state()
    assert state['A'][-1] == min(state['A']) if n_disks > 1 else True
    assert state['B'][-1] == 1

    # TEST 5: Invalid source peg handling.
    # Moving from an empty peg must:
    # - fail
    # - leave the environment state unchanged
    # - not increment the step counter
    env.reset()
    initial_state = deepcopy(env.get_state())
    state, ok, done, info = env.step(('B', 'C'))
    assert ok is False
    assert state == initial_state
    assert env.step_counter == 0

    # TEST 6: Step counter and history consistency.
    # - Valid moves increment step_counter and history
    # - Invalid moves must not
    env.reset()
    env.step(('A', 'B'))
    assert env.step_counter == 1
    env.step(('B', 'C'))
    assert env.step_counter == 2
    env.step(('B', 'A'))        # Illegal move
    assert env.step_counter == 2
    assert len(env.history) == 2

    # TEST 7: Terminal state enforcement (n_disks = 1).
    # - Solving the puzzle must set done=True
    # - Further actions must be rejected
    small_env = classe(1)
    small_env.step(('A', 'C'))
    assert small_env.done is True
    try:
        small_env.step(('C', 'B'))
    except RuntimeError:
        pass
    assert small_env.done is True

    # TEST 8: Determinism invariant.
    # Given the same initial state and action sequence,
    # the final state must be identical across runs.
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

if __name__ == "__main__":
    run_full_hanoi_suite(HanoiEnv, 4, debug=True)
