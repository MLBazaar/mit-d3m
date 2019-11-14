import pathlib
from unittest import TestCase, expectedFailure
from unittest.mock import ANY, patch

from mit_d3m import (
    BUCKET, download_dataset, extract_dataset, get_client, get_dataset_dir, get_dataset_s3_key,
    get_dataset_tarfile_path)


class MitD3mTest(TestCase):

    def test_get_client(self):
        client = get_client()
        self.assertIsNotNone(client)

    def test_get_dataset_tarfile_path(self):
        expected_path = '/foo/bar/things.tar.gz'
        dataset = 'things'

        for datapath in ['/foo/bar', '/foo/bar/', pathlib.Path('/foo', 'bar')]:
            path = get_dataset_tarfile_path(datapath, dataset)
            self.assertEqual(expected_path, path)

    def test_get_dataset_dir(self):
        expected_dir = '/foo/bar/things'
        datapath = '/foo/bar'
        dataset = 'things'
        actual_dir = get_dataset_dir(datapath, dataset)
        self.assertEqual(expected_dir, actual_dir)

    def test_dataset_s3_key(self):
        expected_key = 'datasets/things.tar.gz'
        dataset = 'things'
        actual_key = get_dataset_s3_key(dataset)
        self.assertEqual(expected_key, actual_key)

    @patch('mit_d3m.get_client')
    def test_download_dataset(self, mock_get_client):
        mock_download_file = mock_get_client.return_value.download_file
        bucket = BUCKET
        key = 'things.tar.gz'
        filename = '/foo/bar/things.tar.gz'
        download_dataset(bucket, key, filename)
        mock_download_file.assert_called_with(Bucket=bucket, Key=key, Filename=filename)

    @patch('os.path.exists')
    @patch('tarfile.open')
    def test_extract_dataset(self, mock_open, mock_exists):
        mock_exists.return_value = True
        mock_extractall = mock_open.return_value.__enter__.return_value.extractall
        src = '/foo/bar/things.tar.gz'
        dst = '/foo/bar/things'
        extract_dataset(src, dst)

        mock_open.assert_called_with(src, ANY)
        mock_extractall.assert_called_with(dst)

    @expectedFailure
    def test_load_d3mds(self):
        raise NotImplementedError

    @expectedFailure
    def test_load_dataset(self):
        raise NotImplementedError
