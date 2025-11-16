try:
    raise ValueError("直接抛出")
except Exception as e:
    print(e.__cause__)  # None
    print(e.__context__)  # None

try:
    try:
        raise KeyError("原始错误")
    except KeyError as ke:
        raise RuntimeError("包装错误") from ke
except Exception as e:
    print(e.__cause__)  # KeyError("原始错误")
    print(e.__context__)  # None（因为用了 from，隐式 context 被抑制）

try:
    try:
        raise IndexError("前一个错误")
    except IndexError:
        raise TypeError("新错误")
except Exception as e:
    print(e.__cause__)  # None
    print(e.__context__)  # IndexError("前一个错误")

try:
    raise OSError("原始")
except Exception as e:
    try:
        raise  # 重新抛出 e
    except Exception as e2:
        print(e2 is e)  # True
        print(e2.__cause__)  # None
        print(e2.__context__)  # None
