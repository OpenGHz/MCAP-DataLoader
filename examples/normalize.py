if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from examples.multi_datasets import create_multi_datasets_example
    from mcap_data_loader.pipelines.normalize import Standardize, StandardizeConfig
    from mcap_data_loader.utils.extra_itertools import first_recursive
    from pprint import pprint

    datasets = create_multi_datasets_example()
    pipeline = Standardize(StandardizeConfig(depth=2, strict=False, replace=True))

    print("---- Before Standardization ----")
    before = first_recursive(datasets, depth=3)
    pprint(before)
    after_ds = pipeline(datasets)
    print("---- After Standardization ----")
    after = first_recursive(after_ds, depth=3)
    pprint(after)
    inverse_pipeline = Standardize(StandardizeConfig(inverse=True))
    restored_ds = inverse_pipeline(after_ds)
    print("---- After Inverse Standardization ----")
    restored = first_recursive(restored_ds, depth=3)
    pprint(restored)
