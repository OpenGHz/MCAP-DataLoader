from mcap_data_loader.callers.basis import (
    CallerEnsembleBasis,
    CallerEnsembleConfig,
    ReturnT,
)


class CallerChainConfig(CallerEnsembleConfig, frozen=True):
    """Configuration for CallerChain caller."""

    single_input: bool = False
    """Whether the input to the chain is a single value (slightly faster) or are args & kwargs."""


class CallerChain(CallerEnsembleBasis[ReturnT]):
    """A caller that chains multiple callers together."""

    def __init__(self, config: CallerChainConfig):
        self._callables = config.callables
        self._single_input = config.single_input

    def reset(self):
        self.output_chain = []
        return super().reset()

    def _single_call(self, input):
        output = input
        for caller in self._callables:
            output = caller(output)
            self.output_chain.append(output)
        return output

    def _multi_call(self, *args, **kwds):
        first = True
        for caller in self._callables:
            if first:
                output = caller(*args, **kwds)
                first = False
            else:
                output = caller(output)
            self.output_chain.append(output)
        return output

    def __call__(self, *args, **kwds):
        if self._single_input:
            return self._single_call(*args, **kwds)
        else:
            return self._multi_call(*args, **kwds)


if __name__ == "__main__":
    caller_chain = CallerChain(
        config=CallerChainConfig(callables=[lambda x: x + 1, lambda x: x * 2])
    )
    caller_chain.configure()
    caller_chain.reset()
    print(caller_chain(x=0.0))  # Should print 2.0
    print(caller_chain.output_chain)  # Should print [1.0, 2.0]
    caller_chain.output_chain = []
