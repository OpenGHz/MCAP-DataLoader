class Test:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self._pretuple = (self.a, self.b, self.c)

    @staticmethod
    def _sum(a, b, c):
        return a + b + c

    def sum_normal(self):
        return self._sum(self.a, self.b, self.c)

    def sum_pretuple_star(self):
        return self._sum(*self._pretuple)

    def sum_pretuple_index(self):
        return self._sum(self._pretuple[0], self._pretuple[1], self._pretuple[2])


if __name__ == "__main__":
    import timeit

    test = Test(1, 2, 3)
    rounds = 10_000_000
    print("sum_normal:", timeit.timeit(test.sum_normal, number=rounds))
    print("sum_pretuple_star:", timeit.timeit(test.sum_pretuple_star, number=rounds))
    print("sum_pretuple_index:", timeit.timeit(test.sum_pretuple_index, number=rounds))
