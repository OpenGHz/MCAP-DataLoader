from pydantic import BaseModel, NonNegativeInt
from collections.abc import Generator
from mcap_data_loader.pipelines.basis import Pipeline, T
from typing import Tuple, Dict


Item = Dict[str, T]


class DictTupleConfig(BaseModel):
    depth: NonNegativeInt = 1

    def model_post_init(self, context):
        if self.depth != 1:
            raise NotImplementedError("Only depth=1 is supported currently.")


class DictTuple(Pipeline[Tuple[Item]]):
    """Convert"""

    def __init__(self, config: DictTupleConfig) -> None:
        self.config = config

    def __iter__(self) -> Generator[Item]:
        for item in self._iterables:
            yield self._process(item)

    def _process(self, tp: Tuple[Item]) -> Item:
        tuple_dict = {}
        for i, value in enumerate(tp):
            for k, v in value.items():
                tuple_dict[f"{i}/{k}"] = v
        return tuple_dict


if __name__ == "__main__":
    import time

    tuple_dict = ({"1": 1}, {"2": 2})
    dict_tuple = DictTuple(DictTupleConfig())([tuple_dict])
    start = time.perf_counter()
    result = next(iter(dict_tuple))
    print(f"Time taken: {(time.perf_counter() - start) * 1000:.3f} ms")
    print(result)
