from sacred.observers import MongoObserver, FileStorageObserver
import incense
import torch
import os
import semseg_density.settings as settings


def get_observer():
  if hasattr(settings, 'EXPERIMENT_DB_HOST') and settings.EXPERIMENT_DB_HOST:
    print('mongo observer created', flush=True)
    return MongoObserver.create(url='mongodb://{user}:{pwd}@{host}/{db}'.format(
        host=settings.EXPERIMENT_DB_HOST,
        user=settings.EXPERIMENT_DB_USER,
        pwd=settings.EXPERIMENT_DB_PWD,
        db=settings.EXPERIMENT_DB_NAME),
                                db_name=settings.EXPERIMENT_DB_NAME)
  elif hasattr(settings, 'EXPERIMENT_STORAGE_FOLDER') \
          and settings.EXPERIMENT_STORAGE_FOLDER:
    return FileStorageObserver.create(settings.EXPERIMENT_STORAGE_FOLDER)
  else:
    raise UserWarning("No observer settings found.")


def get_incense_loader():
  if hasattr(settings, 'EXPERIMENT_DB_HOST') and settings.EXPERIMENT_DB_HOST:
    mongouri = 'mongodb://{user}:{pwd}@{host}/{db}'.format(
        host=settings.EXPERIMENT_DB_HOST,
        user=settings.EXPERIMENT_DB_USER,
        pwd=settings.EXPERIMENT_DB_PWD,
        db=settings.EXPERIMENT_DB_NAME)
    return incense.ExperimentLoader(mongo_uri=mongouri,
                                    db_name=settings.EXPERIMENT_DB_NAME)
  else:
    raise UserWarning("No loader settings found.")


def get_checkpoint(pretrained_model, pthname=None):
  # Load pretrained weights
  if pretrained_model and isinstance(pretrained_model, str):
    if '/' in pretrained_model:
      # pretrained model is a filepath
      checkpoint = torch.load(pretrained_model)
      pretrained_id = pretrained_model.split('/')[-1].split('_')[0]
    else:
      # pretrained model is a google drive id
      checkpoint = torch.load(load_gdrive_file(pretrained_model, ending='pth'))
      pretrained_id = pretrained_model
  elif pretrained_model and isinstance(pretrained_model, int):
    loader = get_incense_loader()
    train_exp = loader.find_by_id(pretrained_model)
    if pthname is None:
      print(sorted(list(train_exp.artifacts.keys())))
      pthname = sorted(list(train_exp.artifacts.keys()))[-1]
    train_exp.artifacts[pthname].save(settings.TMPDIR)
    checkpoint = torch.load(
        os.path.join(settings.TMPDIR, f'{pretrained_model}_{pthname}'))
    pretrained_id = str(pretrained_model)
  else:
    assert False
  return checkpoint, pretrained_id
