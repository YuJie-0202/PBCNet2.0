import os



def makedirs(path: str,
             isfile: bool = False) -> None:
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


class Writer:
    def __init__(self, file_path):
        self.file_path = file_path
        makedirs(os.path.dirname(file_path))

    def __call__(self, message):
        with open(self.file_path, "a+") as f:
            f.write(message+"\n")

