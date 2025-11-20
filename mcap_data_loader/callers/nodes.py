from collections.abc import Iterable
from typing import Optional
from pydantic import BaseModel, ConfigDict, NonNegativeInt
from torchdata import nodes
from mcap_data_loader.callers.basis import CallerBasis, ReturnT
from mcap_data_loader.utils.dict import iterable2dict
from toolz import curry


"""Make nodes curried."""
IterableWrapper = curry(nodes.IterableWrapper)
Batcher = curry(nodes.Batcher)
ParallelMapper = curry(nodes.ParallelMapper)
Loader = curry(nodes.Loader)


class MultiNodeWeightedSamplerConfig(BaseModel, frozen=True):
    """Configuration for `MultiNodeWeightedSampler`."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    weights: Iterable[str] = []
    """List of weight keys for each item (dataset/episode)."""
    stop_criteria: str = nodes.StopCriteria.ALL_DATASETS_EXHAUSTED
    """Stop criteria for weighted sampling. Options are defined in `nodes.StopCriteria`."""
    rank: Optional[NonNegativeInt] = None
    """The rank of the current process. Default is None, in which case the rank
    will be obtained from the distributed environment."""
    world_size: Optional[NonNegativeInt] = None
    """The world size of the distributed environment. Default is None, in
    which case the world size will be obtained from the distributed environment."""
    seed: int = 0
    """The seed for the random number generator. Default is 0."""

    def model_post_init(self, context):
        choices = set(nodes.StopCriteria.__dict__.values())
        if self.stop_criteria not in choices:
            raise ValueError(
                f"Invalid stop_criteria: {self.stop_criteria}.Must be one of {choices}."
            )


class MultiNodeWeightedSampler(CallerBasis):
    """Wrap the `nodes.MultiNodeWeightedSampler` into a curried caller."""

    def __init__(self, config: MultiNodeWeightedSamplerConfig):
        self._config_dict = config.model_dump(exclude={"weights"})
        weights = tuple(config.weights)
        self._weights = iterable2dict(weights) if weights else {}

    def __call__(
        self, iterable: Iterable[Iterable[ReturnT]]
    ) -> nodes.MultiNodeWeightedSampler[ReturnT]:
        iterable = tuple(iterable)
        return nodes.MultiNodeWeightedSampler(
            iterable2dict(iterable),
            self._weights or {i: 1.0 for i in range(len(iterable))},
            **self._config_dict,
        )


if __name__ == "__main__":
    from collections import Counter

    wrapper = [IterableWrapper(range(10)) for _ in range(2)]
    sampler = MultiNodeWeightedSampler(MultiNodeWeightedSamplerConfig())
    sampled = sampler(wrapper)

    counter = Counter(sampled)
    most_common = counter.most_common()
    assert most_common[0][1] == most_common[-1][1]
