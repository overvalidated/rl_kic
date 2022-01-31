import numpy as np

class TemplateMask:
    def __init__(self, n_persons, n_shifts, const_hours):
        self.n_persons = n_persons
        self.template = np.zeros((n_persons, n_shifts))
        self.const_hours = const_hours
        self.current_position = 0

        self.work_row = [[[5, 2], [4, 3], [3, 3], [2, 2], None, 2] for i in range(n_persons)]

    def add_next_shift(self, mask):
        assert mask.shape[0] == self.template.shape[0], "wtf dude mask is not the same shape as template"
        self.template[:, self.current_position] = mask * self.const_hours
        self.current_position += 1

        ok = True
        if self.current_position == 1:
            for i in range(self.n_persons):
                self.work_row[i][4] = bool(1 - mask[i])

        for n, el in enumerate(mask):
            # в зависимости от конст часов допустимы разные виды расписания
            # если 7-9 часов, то 5/2 или 4/3
            # если 10 то 4/3 3/3 2/2
            # 11-12 => 3/3 2/2
            reverse = self.work_row[n][4]

            if not reverse:
                if el == 0 and self.work_row[n][-1] == 2:
                    self.work_row[n][-1] -= 1
                elif el == 1 and self.work_row[n][-1] == 1:
                    self.work_row[n][-1] = 0
            else:
                if el == 1 and self.work_row[n][-1] == 2:
                    self.work_row[n][-1] -= 1
                elif el == 0 and self.work_row[n][-1] == 1:
                    self.work_row[n][-1] = 0

            if self.work_row[n][-1] == 0:
                oki = 0
                for i in range(4):
                    if self.work_row[n][i][0] == self.work_row[n][i][1] == 0:
                        oki = i+1
                        break
                
                if 7 <= self.const_hours[n] <= 9:
                    ok = min(ok, oki < 3)
                elif self.const_hours[n] == 10:
                    ok = min(ok, oki > 1)
                elif 11 <= self.const_hours[n] <= 12:
                    ok = min(ok, oki > 2)

                self.work_row[n][:4] = [[5, 2], [4, 3], [3, 3], [2, 2]]
                self.work_row[n][-1] = 2
            
            oki = False
            for i in range(4):
                self.work_row[n][i][1-el] -= 1
                if min(self.work_row[n][i]) >= 0:
                    oki = True
            ok = min(ok, oki)
        return ok

    def calc_current_hours(self):
        return (
            np.sum(self.template[:, self.current_position])
        )

    def calc_summary_hours(self):
        # calculating hours for oso and norm
        return (
            self.template.sum(axis=0),
            self.template.sum(axis=1)
        )

class TemplateMaskController:
    def __init__(self, const_hours, n_persons=8, n_shifts=28):
        self.n_persons = n_persons
        self.n_shifts = n_shifts
        self.hours = np.zeros((n_persons, n_shifts))
        self.const_hours = const_hours
        self.current_position = 0
        self.checker_pos_idx = np.zeros((n_persons, )).astype(np.int32)
        self.mask_possible_templates = np.zeros((n_persons, 4), dtype=np.bool)
        for i in range(n_persons):
            self.mask_possible_templates[i, 2:] += (7 <= self.const_hours[i] <= 9)
            self.mask_possible_templates[i, :3] += (self.const_hours[i] == 10)
            self.mask_possible_templates[i, 0] += (11 <= self.const_hours[i] <= 12)    
        
    def add_shift(self, hours: np.ndarray):
        """
            Adds hours to array and checks for compliability.
        """
        self.hours[:, self.current_position] = hours
        self.current_position += 1
        return self.check_shift()# and not (self.current_position == self.n_shifts)

    def check_shift(self):
        """
            Function to check if hours are compliant with rules;
            checks only last shifts that haven't been seen already
            (as it makes no sense to re-check especially in terms of the problem)

            Checking is done by looking after position and saving possible 
                templates until last added shift is reached; 
                if any template can fit the hours returns True.
            
            Early terminates
        """

        ok = True

        for person_idx in range(self.n_persons):
            if self.hours[person_idx, self.checker_pos_idx[person_idx]] == 0:
                ok = False

            templates = np.array([(2, 2), (3, 3), (4, 3), (5, 2)])
            binary_mask = (self.hours[person_idx, self.checker_pos_idx[person_idx]:self.current_position] > 0).astype(np.int32)
            subt = np.array([[binary_mask.sum(), binary_mask.size - binary_mask.sum()]])
            templates -= subt
            ok = ok and (
                np.max((templates.min(axis=1) >= 0)&(self.mask_possible_templates[person_idx])) and 
                (
                    sorted(binary_mask.tolist()) == list(reversed(binary_mask.tolist()))
                )
            )
            # print(person_idx, binary_mask, templates.ravel())

            if ok and np.max(np.abs(templates).sum(axis=1) == 0):
                self.checker_pos_idx[person_idx] = self.current_position

            if not ok:
                break
        return ok


class Schedule:
    def __init__(self, n_persons=8, n_shifts=42):
        # храним часы и шаблоны
        self.hours = np.zeros((n_persons, n_shifts))

        self.template_mask = TemplateMask(
            n_persons=n_persons, 
            n_shifts=n_shifts
        )

        self.n_persons = n_persons
        self.n_shifts = n_shifts
        
if __name__ == "__main__":
    mask = TemplateMaskController(
        n_persons=3, 
        n_shifts=10, 
        const_hours=np.array([8, 10, 10])
    )
    print(1, mask.add_shift(np.array([1,1,1])))
    print(2, mask.add_shift(np.array([1,1,1])))
    print(3, mask.add_shift(np.array([1,1,1])))
    print(4, mask.add_shift(np.array([0,1,0])))
    print(5, mask.add_shift(np.array([0,1,0])))
    print(6, mask.add_shift(np.array([0,0,0])))
    print(7, mask.add_shift(np.array([1,0,0])))
    print(8, mask.add_shift(np.array([1,1,1])))
    # print(mask.calc_summary_hours())
