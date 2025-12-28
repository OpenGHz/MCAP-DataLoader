import timeit
from dataclasses import dataclass
from pydantic import BaseModel


# 定义普通 dataclass
@dataclass
class DataClassUser:
    name: str
    age: int
    email: str
    is_active: bool


# 定义 Pydantic 模型
class PydanticUser(BaseModel):
    name: str
    age: int
    email: str
    is_active: bool


# 测试数据
test_data = {
    "name": "Alice",
    "age": 30,
    "email": "alice@example.com",
    "is_active": True,
}


# 初始化函数
def create_dataclass():
    return DataClassUser(**test_data)


def create_pydantic():
    return PydanticUser(**test_data)


# 性能测试
def benchmark():
    number = 100_000  # 执行次数

    # 测试 dataclass
    time_dc = timeit.timeit(create_dataclass, number=number)

    # 测试 pydantic
    time_pd = timeit.timeit(create_pydantic, number=number)

    print(f"Dataclass 初始化 {number} 次耗时: {time_dc:.4f} 秒, fps: {number / time_dc:.2f}")
    print(f"Pydantic 初始化 {number} 次耗时: {time_pd:.4f} 秒, fps: {number / time_pd:.2f}")
    print(f"Pydantic 比 Dataclass 慢 {time_pd / time_dc:.2f} 倍")


if __name__ == "__main__":
    benchmark()
