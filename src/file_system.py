"""
FileSystem class.
"""
import os

from s3fs import S3FileSystem


class FileSystem:
    """
    FileSystem class which allows to get a connection to MinIO.
    """

    def __init__(self):
        """
        Constructor for the FileSystem class.
        """
        self.endpoint = os.environ["AWS_S3_ENDPOINT"]
        self.key = os.environ["AWS_ACCESS_KEY_ID"]
        self.secret = os.environ["AWS_SECRET_ACCESS_KEY"]
        self.token = os.environ["AWS_SESSION_TOKEN"]

    def get_file_system(self) -> S3FileSystem:
        """
        Returns the s3 file system.
        """
        return S3FileSystem(
            client_kwargs={"endpoint_url": "https://" + self.endpoint},
            key=self.key,
            secret=self.secret,
            token=self.token,
        )
