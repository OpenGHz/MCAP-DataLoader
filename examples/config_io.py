if __name__ == "__main__":
    from mcap_data_loader.basis.cfgable import InitConfigMixin
    from pydantic import BaseModel

    class MyConfig(BaseModel):
        param1: int
        param2: str

    class MyClass(InitConfigMixin):
        def __init__(self, config: MyConfig):
            self.config = config

    config = MyConfig(param1=10, param2="example")
    my_instance = MyClass(config)
    my_instance.save_config("my_config.yaml")

    loaded_instance = MyClass("my_config.yaml")
    assert loaded_instance.config == my_instance.config
    print("Configuration saved and loaded successfully.")

    # dict input
    instance = MyClass(config.model_dump())
    assert instance.config == my_instance.config
    # only kwargs input
    instance = MyClass(param1=config.param1, param2=config.param2)
    assert instance.config == my_instance.config
    # config with kwargs
    instance = MyClass(config, param2="modified")
    assert instance.config.param2 == "modified"

    from dataclasses import dataclass, asdict

    @dataclass
    class MyDataClassConfig:
        paramA: float
        paramB: str

    class MyDataClass(InitConfigMixin):
        def __init__(self, config: MyDataClassConfig):
            self.config = config

    dataclass_config = MyDataClassConfig(paramA=3.14, paramB="data")
    data_instance = MyDataClass(dataclass_config)
    assert data_instance.config == dataclass_config
    data_instance = MyDataClass(asdict(dataclass_config))
    assert data_instance.config == dataclass_config
    data_instance = MyDataClass(paramA=2.71, paramB="data2")
    assert data_instance.config.paramA == 2.71
    data_instance = MyDataClass(dataclass_config, paramB="modified2")
    assert data_instance.config.paramB == "modified2"
    print("Dataclass configuration handling successful.")

    data_instance.save_config("my_config.yaml")
    loaded_data_instance = MyDataClass("my_config.yaml")
    assert loaded_data_instance.config == data_instance.config
    print("Dataclass configuration saved and loaded successfully.")
