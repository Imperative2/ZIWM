class FileWriter:

    path = None
    writer = None

    def __init__(self, path):
        self.path = path
        self.writer = open("test_results.txt", mode="a+")

    def write(self, res_str):
        self.writer.write(res_str)
        self.writer.flush()

    def close(self):
        self.writer.close()