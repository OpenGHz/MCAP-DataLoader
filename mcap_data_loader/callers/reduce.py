from pydantic import BaseModel, PositiveInt
from mcap_data_loader.callers.basis import CallerBasis
from mcap_data_loader.utils.array_like import Array


class FrequencyReductionCallConfig(BaseModel, frozen=True):
    # 1 means call at each step
    period: PositiveInt = 1


class FrequencyReductionCall(CallerBasis[Array]):
    def __init__(self, config: FrequencyReductionCallConfig):
        self._period = config.period
        self.reset()

    def check_shape(self, output: Array):
        horizon = output.shape[1]
        period = self._period
        if horizon < period:
            raise ValueError(
                f"output horizon: {horizon} can not be shorter than period: {period}"
            )
        return output

    def reset(self):
        self._t = 0
        self._first = True

    def __call__(self, array: Array):
        if self._first:
            self._outputs = self.check_shape(array)
            self._first = False
        target_t = self._t % self._period
        if target_t == 0:
            self._outputs = array
        self._t += 1
        # TODO: allow configure the t position
        return self._outputs[:, target_t]
