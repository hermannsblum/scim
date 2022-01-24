import torch
import tensorflow_datasets as tfds

class TFDSIterableDataset(torch.utils.data.IterableDataset):
  def __init__(self, *args, **kwargs):
    super().__init__()
    self.tfds_data = tfds.load(*args, **kwargs)

  def __iter__(self):
    return iter(tfds.as_numpy(self.tfds_data))

  def __len__(self):
    return len(self.tfds_data)

class TFDataIterableDataset(torch.utils.data.IterableDataset):
  def __init__(self, ds):
    super().__init__()
    self.tf_dataset = tfds.as_numpy(ds)

  def __iter__(self):
      for batch in self.tf_dataset:
          yield torch.from_numpy(batch[0]), torch.from_numpy(batch[1])

  def __len__(self):
    return len(self.tf_dataset)

class FilteredTFDataIterableDataset(torch.utils.data.IterableDataset):
  def __init__(self, ds, remove_classes):
    """
    Args:
        ds: a tf.data.Dataset
        remove_classes: a numpy vector of shape [max_classes] mapping to
            0 (ok) or 1 (datapoint should be removed)
            Any datapoint with sum > 0 will be removed.
    """
    super().__init__()
    self.tf_dataset = tfds.as_numpy(ds)
    self.remove_classes = remove_classes


  def __iter__(self):
      for x, y in self.tf_dataset:
          if np.sum(self.remove_classes[y]) > 0:
            continue
          yield torch.from_numpy(x), torch.from_numpy(y)

  def __len__(self):
    return len(self.tf_dataset)
