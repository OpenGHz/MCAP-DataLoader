from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersSampleDataset,
    McapDatasetConfig,
)
from pprint import pprint


path = "data/example"
dataset = McapFlatBuffersSampleDataset(
    McapDatasetConfig(
        data_root=path,
        topics=[
            "/follow/arm/pose/position",
        ],
    )
)
dataset.load()

# for index, sample in enumerate(dataset.reader.iter_attachment_samples(color_topics)):
#     # print(f"Sample {index}: {sample.keys()}")
#     # print(index)
#     pass

for index, sample in enumerate(dataset):
    pprint(sample)
