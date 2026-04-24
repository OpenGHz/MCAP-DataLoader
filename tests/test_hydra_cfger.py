import os
import sys

from pydantic import BaseModel

from mcap_data_loader.configurers.hydra_cfger import Configurer


class DemoConfig(BaseModel):
    foo: int = 0


def test_single_run_resolves_hydra_job_env_set(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "foo: 7",
                "hydra:",
                "  job:",
                "    env_set:",
                "      CUDA_VISIBLE_DEVICES: ${floor_div:${hydra:job.num},2}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest", "--config-path", str(config_path)])

    configurer = Configurer(DemoConfig)
    configurer.parse()

    cfg = configurer.configure()

    assert cfg.foo == 7
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
