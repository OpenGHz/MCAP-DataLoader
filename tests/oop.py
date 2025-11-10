import timeit
from dataclasses import dataclass
from pydantic import BaseModel


@dataclass(slots=True)
class ConfigDCS:
    item0: int = 0
    item1: int = 1
    item2: int = 2


@dataclass()
class ConfigDC:
    item0: int = 0
    item1: int = 1
    item2: int = 2


class ConfigBM(BaseModel):
    item0: int = 0
    item1: int = 1
    item2: int = 2


class ConfigBMF(BaseModel, frozen=True):
    item0: int = 0
    item1: int = 1
    item2: int = 2


config_dcs = ConfigDCS()
config_dc = ConfigDC()
config_bm = ConfigBM()
config_bms = ConfigBMF()
lis3 = [1, 2, 3]


def for_list():
    a = lis3[0]
    b = lis3[1]
    c = lis3[2]


def for_config(config):
    a = config.item0
    b = config.item1
    c = config.item2


def for_config_dcs():
    for_config(config_dcs)


def for_config_dc():
    for_config(config_dc)


def for_config_bm():
    for_config(config_bm)


def for_config_bms():
    for_config(config_bms)


if __name__ == "__main__":
    rounds = 10000000
    print("for_list:", timeit.timeit(for_list, number=rounds))
    print("for_config_dcs:", timeit.timeit(for_config_dcs, number=rounds))
    print("for_config_dc:", timeit.timeit(for_config_dc, number=rounds))
    print("for_config_bmf:", timeit.timeit(for_config_bms, number=rounds))
    print("for_config_bm:", timeit.timeit(for_config_bm, number=rounds))
