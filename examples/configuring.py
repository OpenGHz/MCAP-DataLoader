if __name__ == "__main__":
    import logging
    from pydantic import BaseModel, ConfigDict
    from mcap_data_loader.basis.cfgable import ConfigurableBasis

    logging.basicConfig(level=logging.INFO)

    class MyConfig(BaseModel):
        """Example configuration model."""

        param1: int = 10
        param2: str = "default"

    class MyComponent(ConfigurableBasis):
        def __init__(self, config: MyConfig):
            self.config = config

        def on_configure(self) -> bool:
            self.get_logger().info(f"Configuring with {self.config}")
            return True

    class MyCompsConfig(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        comps: list[MyComponent]

    comp = MyComponent(param1=20)
    assert comp.configure()
    assert comp.all_configure()
    print(comp.dump())
    comp.copy()
    comp.copy(True)
    comps_config = MyCompsConfig(comps=[comp])
    print(comps_config)
    print(comps_config.model_dump(mode="json", fallback=lambda x: x.dump()))
