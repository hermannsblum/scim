
import os
import gdrive

def load_gdrive_file(file_id,
                     ending='',
                     output_folder=os.path.expanduser('~/.keras/datasets')):
    """Downloads files from google drive, caches files that are already downloaded."""
    filename = '{}.{}'.format(file_id, ending) if ending else file_id
    filename = os.path.join(output_folder, filename)
    if not os.path.exists(filename):
        gdown.download('https://drive.google.com/uc?id={}'.format(file_id),
                       filename,
                       quiet=False)
    return filename
