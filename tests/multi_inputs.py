import timeit
from pydantic import BaseModel


lis3 = [1, 2, 3]
lis4 = [0, 1, 2, 3]
tuple3 = tuple(lis3)


def sum4(a, b, c, d):
    return b + c + d


direct = lambda lis: sum(lis)


def t0():
    lis4[0] = 5
    return sum(lis4)


def t1():
    lis4[0] = 5
    return direct(lis4)


def t2():
    return direct((0,) + tuple3)


def t3():
    b, c, d = tuple3
    return direct((5, b, c, d))


def t4():
    return direct((0, *tuple3))


def test0():
    return sum4(0, 1, 2, 3)


def test1():
    return sum4(*lis4)


def test2():
    b, c, d = lis3
    return sum4(0, b, c, d)


def test3():
    lis4[0] = 5
    return sum4(*lis4)


def test4():
    return sum4(0, lis3[0], lis3[1], lis3[2])


def test5():
    return sum4(0, *lis3)


def test6():
    return sum4(*([0] + lis3))


if __name__ == "__main__":
    rounds = 10000000
    print("t0:", timeit.timeit(t0, number=rounds))
    print("t1:", timeit.timeit(t1, number=rounds))
    print("t2:", timeit.timeit(t2, number=rounds))
    print("t3:", timeit.timeit(t3, number=rounds))
    print("t4:", timeit.timeit(t4, number=rounds))
    print("test0:", timeit.timeit(test0, number=rounds))
    print("test1:", timeit.timeit(test1, number=rounds))
    print("test2:", timeit.timeit(test2, number=rounds))
    print("test3:", timeit.timeit(test3, number=rounds))
    print("test4:", timeit.timeit(test4, number=rounds))
    print("test5:", timeit.timeit(test5, number=rounds))
    print("test6:", timeit.timeit(test6, number=rounds))
