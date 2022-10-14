"""
FileSystem class.
"""
import os

from s3fs import S3FileSystem


class FileSystem:
    """
    FileSystem class which allows to get a connection to MinIO.
    """

    def __init__(
        self,
        endpoint: str = os.environ["AWS_S3_ENDPOINT"],
        key: str = os.environ["AWS_ACCESS_KEY_ID"],
        secret: str = os.environ["AWS_SECRET_ACCESS_KEY"],
    ):
        """
        Constructor for the FileSystem class.
        """
        self.endpoint = endpoint
        self.key = key
        self.secret = secret

    def get_file_system(self) -> S3FileSystem:
        """
        Returns the s3 file system.
        """
        return S3FileSystem(
            client_kwargs={"endpoint_url": "https://" + self.endpoint},
            key=self.key,
            secret=self.secret,
        )
