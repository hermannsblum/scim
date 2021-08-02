from sacred.observers import MongoObserver, FileStorageObserver
import incense
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
