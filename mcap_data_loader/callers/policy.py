from pydantic import BaseModel, NonNegativeInt
from typing import Any
from mcap_data_loader.callers.basis import CallerBasis, ReturnT


class PolicyEvaluationCallerConfig(BaseModel, frozen=True):
    """Configuration for policy evaluation."""

    # max number of evaluation steps per rollout, 0 means no limit
    max_steps: NonNegativeInt = 0
    # number of rollouts to evaluate, 0 means infinite
    num_rollouts: NonNegativeInt = 0
    # checkpoint path to load the policy from
    checkpoint_path: str = ""
    # method name to call the policy, empty means `__call__`
    call_method_name: str = ""


class PolicyEvaluationCaller(CallerBasis[ReturnT]):
    config: PolicyEvaluationCallerConfig

    def on_configure(self):
        return True

    def reset(self):
        """Reset the internal state of the caller, if any."""

    def __call__(self, *args, **kwds) -> Any:
        """Call the caller with the given inputs."""
