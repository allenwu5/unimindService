class CifarImage:
    def __init__(self):
        self.name = 'undefined cifar image name'
        self.path = ''
        self.body = '' # IO steam

    @property
    def content(self):
        content = self.body if self.body else self.path
        assert content # must be exist
        return content
