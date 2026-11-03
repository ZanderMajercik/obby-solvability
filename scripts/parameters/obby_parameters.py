import inspect
import shutil

class ObbyParameters:
    def get_generators(self):
        pass

    def save(self, path):
        file_path = inspect.getfile(self.__class__)
        print(f"Loaded params from {file_path}, saving to {path}")
        shutil.copy(file_path, path)
        return



