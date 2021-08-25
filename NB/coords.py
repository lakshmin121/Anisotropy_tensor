import numpy as np

class coord:
    """
    Class to create coordinate variables. Stores coordinates of a point in 3D space as a tuple.
    Operations like vector addition of coordinates or finding distance between them is implemented.

    Doesn't distinguish between different order of axes: (x, y, z) or (z, y, x).
    """
    def __init__(self, tuple_like):
        if not type(tuple_like) == tuple:
            tuple_like = tuple(tuple_like)
        self.value = tuple_like

    def __len__(self):
        return len(self.value)

    def __repr__(self):
        return 'coord '+str(self.value)

    def __add__(self, other):
        if isinstance(other, coord):
            return coord(np.asarray(self.value) + np.asarray(other.value))
        elif isinstance(other, float) or isinstance(other, int):
            return coord(np.asarray(self.value) + other)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, coord):
            return coord(np.asarray(self.value) - np.asarray(other.value))
        elif isinstance(other, float) or isinstance(other, int):
            return coord(np.asarray(self.value) - other)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, coord):
            return coord(np.asarray(self.value) * np.asarray(other.value))
        elif isinstance(other, float) or isinstance(other, int):
            return coord(other * np.asarray(self.value))
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __eq__(self, other):
        return self.value == other.value

    def __iter__(self):
        return iter(self.value)

    def __ceil__(self):
        return coord(np.ceil(np.asarray(self.value)))

    def __floor__(self):
        return coord(np.floor(np.asarray(self.value)))

    def dist(self):
        return np.linalg.norm(self.value)


# # Test for coords
# c1 = coord((1, 2, 3))
# c2 = coord((4, 5, 6))
# print("coordinate variables c1 and c2: ", c1, c2)
# print("len(c1): ", len(c1))
# print("c1 + c2 = ", c1 + c2)
# print("1 + c2 = ", c2+1)
# print("c1 * c2 = ", c1*c2)
# print('3 * c1 = ', 3*c1)