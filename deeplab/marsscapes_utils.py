import torch

def data_converter_marsscapes(label):
  label = torch.where(label == torch.tensor(7, dtype=torch.int64), torch.tensor(255, dtype=torch.int64), label)
  label = torch.where(label == torch.tensor(5, dtype=torch.int64), torch.tensor(255, dtype=torch.int64), label)
  label = torch.where(label == torch.tensor(6, dtype=torch.int64), torch.tensor(5, dtype=torch.int64), label)
  label = torch.where(label == torch.tensor(4, dtype=torch.int64), torch.tensor(10, dtype=torch.int64), label)
  label = torch.where(label == torch.tensor(3, dtype=torch.int64), torch.tensor(15, dtype=torch.int64), label)
  label = torch.where(label == torch.tensor(2, dtype=torch.int64), torch.tensor(6, dtype=torch.int64), label)
  label = torch.where(label == torch.tensor(1, dtype=torch.int64), torch.tensor(8, dtype=torch.int64), label)
  label = torch.where(label == torch.tensor(0, dtype=torch.int64), torch.tensor(0, dtype=torch.int64), label)
  return label