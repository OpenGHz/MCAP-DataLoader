import timeit


lis3 = [1, 2, 3]
lis4 = [0, 1, 2, 3]


def sum4(a, b, c, d):
    return a + b + c + d


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
    print("test0:", timeit.timeit(test0, number=rounds))
    print("test1:", timeit.timeit(test1, number=rounds))
    print("test2:", timeit.timeit(test2, number=rounds))
    print("test3:", timeit.timeit(test3, number=rounds))
    print("test4:", timeit.timeit(test4, number=rounds))
    print("test5:", timeit.timeit(test5, number=rounds))
    print("test6:", timeit.timeit(test6, number=rounds))
