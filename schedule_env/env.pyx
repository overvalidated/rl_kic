import numpy as np
import torch
#cimport torch
cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float32

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float32_t DTYPE_t

def new_state(int n_persons, int n_shifts):
    cdef np.ndarray[DTYPE_t, ndim=3] field = np.zeros((n_persons, n_shifts, 15), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] hours = np.zeros((n_persons, n_shifts), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] possible_moves = np.ones((n_persons, n_shifts, 15), dtype=DTYPE) # masking out impossible moves
    cdef np.ndarray[DTYPE_t, ndim=1] worked_days = np.zeros((n_persons, ), dtype=DTYPE)
    cdef int person = 0
    cdef int shift = 0

    return dict(
        field=field,
        hours=hours,
        possible_moves=possible_moves,
        worked_days=worked_days,
        person=person,
        shift=shift,
        n_persons=n_persons,
        n_shifts=n_shifts
    )


def check_mode(int n, int mode, int shift):
    return np.roll(np.array([1, 0] * (mode // 10) + [0, 0] * (mode % 10)), shift*2)[n % ((mode // 10 + mode % 10)*2)]


def prepare_env(
    float[:,:,:] field, 
    float[:,:] hours, 
    float[:,:,:] possible_moves, 
    float[:] worked_days, 
    int person, 
    int shift, 
    int n_persons, 
    int n_shifts,
    int shift_prep):

    state = dict(
        field=field,
        hours=hours,
        possible_moves=possible_moves,
        worked_days=worked_days,
        person=person,
        shift=shift,
        n_persons=n_persons,
        n_shifts=n_shifts
    )

    # 0 4 7 7.5 8 8.5 9 9.5 10 10.5 11 11.5 12 12.5 13

    chosen_modes = []
    shifts = []
    for person_ in range(n_persons):
        chosen_modes += [np.random.choice([22, 33, 43, 52], p=[0.1, 0.2, 0.2, 0.5])]
        if chosen_modes[-1] == 22:
            shifts += [np.random.randint(4)]
        elif chosen_modes[-1] == 33:
            shifts += [np.random.randint(6)]
        elif chosen_modes[-1] == 43:
            shifts += [np.random.randint(-1, 3)]
        elif chosen_modes[-1] == 52:
            shifts += [0]  #[np.random.randint(-1, 2)]
    # shifts = [0,0,2,2,0,0,2,2]

    mids = [0.05, 0.2, 0.5, 0.2, 0.05]
    longs = [0.1, 0.3, 0.6]

    for n in range(shift_prep):
        if n % 2 == 0:
            for mode, shift_ in zip(chosen_modes, shifts):
                if check_mode(n, mode, shift_):
                    if mode == 52:
                        state, _ = next_state(**state, action=np.random.choice([2,3,4,5,6], p=mids))  #7 - 9
                    elif mode == 43:
                        state, _ = next_state(**state, action=np.random.choice([6,7,8], p=longs))  #9, 9.5, 10
                    elif mode == 33:
                        state, _ = next_state(**state, action=np.random.choice([6,7,8], p=longs))  #9, 9.5, 10
                    else:
                        state, _ = next_state(**state, action=np.random.choice([6,7,8], p=longs))  #9, 9.5, 10
                else:
                    state, _ = next_state(**state, action=0)
        else:
            for _ in range(n_persons):
                state, _ = next_state(**state, action=0)

    return state


def is_terminal(
    float[:,:,:] field, 
    float[:,:] hours, 
    float[:,:,:] possible_moves, 
    float[:] worked_days, 
    int person, 
    int shift, 
    int n_persons, 
    int n_shifts):
    return shift == n_shifts


def is_terminal_strict(
    float[:,:,:] field, 
    float[:,:] hours, 
    float[:,:,:] possible_moves, 
    float[:] worked_days, 
    int person, 
    int shift, 
    int n_persons, 
    int n_shifts):
    if shift == n_shifts:
        return True

    cdef int weeks_eval = shift / 14;
    
    # TODO: inefficient
    cdef float sum_over_week = 0;
    cdef Py_ssize_t person_, week, i
    for person_ in range(n_persons):
        for week in range(weeks_eval):
            sum_over_week = 0;
            for i in range(14 + week * 14):
                sum_over_week += hours[person][shift+i]
            
            if ((sum_over_week > 42) or (sum_over_week<35)):
                return True

    return False


def get_possible_moves(
    float[:,:,:] field, 
    float[:,:] hours, 
    float[:,:,:] possible_moves, 
    float[:] worked_days, 
    int person, 
    int shift, 
    int n_persons, 
    int n_shifts):
    cdef float[:] possible_moves_ = possible_moves[person][shift]

    if (shift >= 2):
        if (hours[person][shift-2] > 10):
            return np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=DTYPE)

    # 10 00 0
    if (shift >= 4):
        if ((10 >= hours[person][shift-4] > 0) and
            (hours[person][shift-2] == 0) and
            (hours[person][shift-1] == 0)):
            return np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=DTYPE)

    # 10 00 00
    if (shift >= 5):
        if ((10 >= hours[person][shift-5] > 0) and 
            (hours[person][shift-3] == 0) and
            (hours[person][shift-2] == 0) and
            (hours[person][shift-1] == 0)):
            return np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=DTYPE)
    # 10 10 10 10 10 10
    if (shift >= 10) and (10-np.sum(field[person, shift-10:, 0])>=5):
        return np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=DTYPE)
    
    return possible_moves_;


def get_return(
    float[:,:,:] field, 
    float[:,:] hours, 
    float[:,:,:] possible_moves, 
    float[:] worked_days, 
    int person, 
    int shift, 
    int n_persons, 
    int n_shifts):

    cdef Py_ssize_t i
    cdef int sum_over_week = 0
    # if shift % 14 == 13:
    # sum_over_week = np.sum(hours[person, max(0, shift-13):shift+1])
    # if ((sum_over_week > 42) or (sum_over_week<35)):
    ## check every week and make sure that 
    cdef float reward = 0
    # for each week independently
    # cdef int rew = np.sum(hours[person, 14*(shift//14):14*(shift//14 + 1)]) - 40
    cdef int rew = np.sum(hours[person, max(0, shift-13): shift+1]) - 40
    #if rew == 0:
    #    reward = 2
    #elif rew == 1 or rew == -1 or rew == -2:
    #    reward = 1.5
    #elif rew == 2 or -5 <= rew <= -3:
    #    reward = 1
    #else:
    #    reward = 0.5 - np.abs(rew) / 80 
    #return reward # if shift >= 13 else 0

    # using exponential distribution
    # if rew >= 0:
    #     return np.exp(1.23 * -rew)
    # elif rew < 0:
    #     return np.exp(0.6148 * rew)

    # simple handmade reward
    return rew


def next_state(
    float[:,:,:] field, 
    float[:,:] hours, 
    float[:,:,:] possible_moves, 
    float[:] worked_days, 
    int person, 
    int shift, 
    int n_persons, 
    int n_shifts,
    int action):

    cdef np.ndarray hours_ = np.array([0, 4, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13], dtype=np.float32)

    field[person][shift][action] = 1;
    hours[person][shift] = hours_[action];

    if ((action > 0) and (action < 10)):
        # out of bounds check
        if (shift+1 < n_shifts):
            possible_moves[person][shift+1][1:] = 0
        if (shift+3 < n_shifts):
            possible_moves[person][shift+3][1:] = 0
    elif (action >= 10):
        if (shift+1 < n_shifts):
            possible_moves[person][shift+1][1:] = 0
        
        if (shift+2 < n_shifts):
            possible_moves[person][shift+2][-5:] = 0
        
        if (shift+3 < n_shifts):
            possible_moves[person][shift+3][1:] = 0
    reward = get_return(field, hours, possible_moves, worked_days, person, shift, n_persons, n_shifts)
    person += 1;
    shift = shift + person / n_persons;
    person %= n_persons;
    return dict(
        field=field,
        hours=hours,
        possible_moves=possible_moves,
        worked_days=worked_days,
        person=person,
        shift=shift,
        n_persons=n_persons,
        n_shifts=n_shifts
    ), reward


def stringify_state(
    float[:,:,:] field, 
    float[:,:] hours, 
    float[:,:,:] possible_moves, 
    float[:] worked_days, 
    int person, 
    int shift, 
    int n_persons, 
    int n_shifts):

    cdef str str_ = ""
    cdef Py_ssize_t i = 0 , j = 0
    for i in range(n_persons):
        for j in range(n_shifts):
            str_ += str(hours[i][j]) + '|'
        str_ += '\n';
    return str_;


def get_observation(
    float[:,:,:] field, 
    float[:,:] hours, 
    float[:,:,:] possible_moves, 
    float[:] worked_days, 
    int person, 
    int shift, 
    int n_persons, 
    int n_shifts,
    int shift_pos):

    # int *** observation = new int**[n_persons]; //(n_persons, vector<vector<int>>(n_shifts, vector<int>(11)));
    cdef np.ndarray[DTYPE_t, ndim=2] observation = np.zeros((n_persons, 10), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] target_observation
    #cdef Py_ssize_t person_=0, shift_=0, hour_=0
    #for person_ in range(n_persons):
    #    for shift_ in range(n_shifts):
    #        for hour_ in range(1, 10):
    #            if (field[person_][shift_][hour_] == 1):
    #                observation[person_][shift_][hour_-1] = 1;
    #                break
    observation[:, -min(10, shift_pos):] = hours[:, max(shift_pos-10, 0):shift_pos]
    target_observation = np.reshape(observation, (-1, ))

    cdef np.ndarray[DTYPE_t, ndim=1] position = np.zeros((n_persons, ), dtype=DTYPE)
    position[person] = 1
    target_observation = np.concatenate([np.ravel(target_observation), position])
    return target_observation
