import pyximport 
import numpy as np
from copy import copy, deepcopy
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
import env
import time

def copy_schedule(schedule):
    return dict(
        field=copy(np.array(schedule['field'])),
        hours=copy(np.array(schedule['hours'])),
        possible_moves=copy(np.array(schedule['possible_moves'])),
        worked_days=copy(np.array(schedule['worked_days'])),
        moves_set=copy(np.array(schedule['moves_set'])),
        person=schedule['person'],
        shift=schedule['shift'],
        n_persons=schedule['n_persons'],
        n_shifts=schedule['n_shifts']
    )

if __name__ == "__main__":
    schedule_ = env.new_state(10, 10)
    env.next_state(**copy_schedule(schedule_), action=1)
    print(np.array(schedule_['hours']))
    # print(env.get_return(**schedule_))
