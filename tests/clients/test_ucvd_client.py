from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, call, create_autospec, patch

import pytest
import responses
from requests import HTTPError, Request, Response

from pysolotools.clients.ucvd_client import UCVDClient
from pysolotools.core import UCVDArchive, UCVDAttachment, UCVDDataset
from pysolotools.core.exceptions import UCVDException

MOCK_PROJECT_ID = "mock-proj-id"
MOCK_ORG_ID = "mock-org-id"
MOCK_SA_KEY = "mock-sa-key"
MOCK_API_SECRET = "mock-api-secret"
MOCK_RUN_ID = "123"
BASE_URI = "some-uri"
API_ENDPOINT = f"{BASE_URI}/organizations/{MOCK_ORG_ID}/projects/{MOCK_PROJECT_ID}"

mock_make_request_response = {"results": "the results"}

mock_dataset = {
    "id": "b3eb0caf-52dd-46e5-bf61-ae46bb597e0e",
    "name": "Test Dataset 1",
    "description": "Mock datasets",
    "licenseURI": "string",
    "createdAt": "2022-05-04T07:36:52.233871Z",
    "updatedAt": "2022-05-04T07:36:52.233871Z",
}

mock_list_datasets = {"next": "mock-page-2", "results": [mock_dataset]}

mock_dataset_archive = {
    "id": "id-1",
    "name": "Sample.tar",
    "type": "FULL",
    "downloadURL": "https://mock-signed-url",
    "state": {"status": "READY"},
}

mock_dataset_attachment = {
    "id": "id-1",
    "name": "Sample.tar",
    "downloadURL": "https://mock-signed-url",
    "state": {"status": "READY"},
}


@dataclass
class TestFixture:
    client: UCVDClient
    mock_make_request: Mock


@pytest.fixture
@patch(
    "pysolotools.clients.ucvd_client.UCVDClient._UCVDClient__make_request",
    autospec=True,
)
def setup_client(mock_make_request):
    client = UCVDClient(
        project_id=MOCK_PROJECT_ID,
        org_id=MOCK_ORG_ID,
        sa_key=MOCK_SA_KEY,
        api_secret=MOCK_API_SECRET,
        base_uri=BASE_URI,
    )

    mock_make_request.return_value = mock_make_request_response
    client._UCVDClient__make_request = mock_make_request
    return TestFixture(client=client, mock_make_request=mock_make_request)


@pytest.mark.parametrize(
    "description,license_uri,expected",
    [
        (
            None,
            None,
            {
                "method": "post",
                "url": "some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets",
                "body": {"name": "some-dataset"},
            },
        ),
        (
            None,
            "a-license",
            {
                "method": "post",
                "url": "some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets",
                "body": {"name": "some-dataset", "licenseURI": "a-license"},
            },
        ),
        (
            "a-description",
            None,
            {
                "method": "post",
                "url": "some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets",
                "body": {"name": "some-dataset", "description": "a-description"},
            },
        ),
    ],
)
def test_create_dataset(setup_client: TestFixture, description, license_uri, expected):
    assert (
        setup_client.client.create_dataset(
            dataset_name="some-dataset",
            description=description,
            license_uri=license_uri,
        )
        == mock_make_request_response
    )
    setup_client.mock_make_request.assert_called_once_with(
        **expected, auth=setup_client.client.auth
    )


def test_describe_dataset(setup_client: TestFixture):
    assert (
        setup_client.client.describe_dataset(dataset_id="some-dataset")
        == mock_make_request_response
    )
    setup_client.mock_make_request.assert_called_once_with(
        method="get",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets/some-dataset",
        auth=setup_client.client.auth,
    )


def test_delete_dataset(setup_client: TestFixture):
    assert (
        setup_client.client.delete_dataset(dataset_id="some-dataset")
        == mock_make_request_response
    )
    setup_client.mock_make_request.assert_called_once_with(
        method="delete",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets/some-dataset",
        auth=setup_client.client.auth,
    )


def test_list_dataset(setup_client: TestFixture):
    assert setup_client.client.list_datasets() == "the results"
    setup_client.mock_make_request.assert_called_once_with(
        method="get",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets",
        auth=setup_client.client.auth,
    )


@pytest.mark.parametrize(
    "arguments,expected_archive_name",
    [
        ({"dataset_id": "some-dataset"}, "Archive.tar"),
        (
            {"dataset_id": "some-dataset", "archive_name": "some-archive"},
            "some-archive",
        ),
    ],
)
def test_create_dataset_archives(
    setup_client: TestFixture, arguments, expected_archive_name
):
    assert (
        setup_client.client.create_dataset_archive(**arguments)
        == mock_make_request_response
    )
    setup_client.mock_make_request.assert_called_once_with(
        method="post",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets/some-dataset/archives",
        body={"name": expected_archive_name},
        auth=setup_client.client.auth,
    )


def test_list_dataset_archive(setup_client: TestFixture):
    assert (
        setup_client.client.list_dataset_archives(dataset_id="some-dataset")
        == "the results"
    )
    setup_client.mock_make_request.assert_called_once_with(
        method="get",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets/some-dataset/archives",
        auth=setup_client.client.auth,
    )


def test_describe_dataset_archive(setup_client: TestFixture):
    assert (
        setup_client.client.describe_dataset_archive(
            dataset_id="some-dataset", archive_id="some-archive"
        )
        == mock_make_request_response
    )
    setup_client.mock_make_request.assert_called_once_with(
        method="get",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets/some-dataset/archives/some-archive",
        auth=setup_client.client.auth,
    )


def test_list_dataset_attachments(setup_client: TestFixture):
    assert (
        setup_client.client.list_dataset_attachments(dataset_id="some-dataset")
        == "the results"
    )
    setup_client.mock_make_request.assert_called_once_with(
        method="get",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets/some-dataset/attachments",
        auth=setup_client.client.auth,
    )


def test_describe_dataset_attachment(setup_client: TestFixture):
    assert (
        setup_client.client.describe_dataset_attachment(
            dataset_id="some-dataset", attachment_id="some-attachment"
        )
        == mock_make_request_response
    )
    setup_client.mock_make_request.assert_called_once_with(
        method="get",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets/some-dataset/attachments/some-attachment",
        auth=setup_client.client.auth,
    )


def test_create_dataset_attachment(setup_client: TestFixture):
    assert (
        setup_client.client.create_dataset_attachment(
            dataset_id="some-dataset",
            attachment_name="some-attachment",
            description="some-description",
        )
        == mock_make_request_response
    )
    setup_client.mock_make_request.assert_called_once_with(
        method="post",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets/some-dataset/attachments",
        body={"name": "some-attachment", "description": "some-description"},
        auth=setup_client.client.auth,
    )


def test_create_build(setup_client: TestFixture):
    assert (
        setup_client.client.create_build(
            build_name="some-build", description="some-description"
        )
        == mock_make_request_response
    )
    setup_client.mock_make_request.assert_called_once_with(
        method="post",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/builds",
        body={"name": "some-build", "description": "some-description"},
        auth=setup_client.client.auth,
    )


def test_list_builds(setup_client: TestFixture):
    assert setup_client.client.list_builds() == "the results"
    setup_client.mock_make_request.assert_called_once_with(
        method="get",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/builds",
        auth=setup_client.client.auth,
    )


def test_describe_build(setup_client: TestFixture):
    assert (
        setup_client.client.describe_build(build_id="some-build")
        == mock_make_request_response
    )
    setup_client.mock_make_request.assert_called_once_with(
        method="get",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/builds/some-build",
        auth=setup_client.client.auth,
    )


def test_create_job(setup_client: TestFixture):
    arguments = {
        "name": "some-job",
        "description": "some-description",
        "specs": {"some": "things"},
    }
    assert setup_client.client.create_job(**arguments) == mock_make_request_response
    setup_client.mock_make_request.assert_called_once_with(
        method="post",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/jobs/",
        body={
            "name": arguments.get("name"),
            "description": arguments.get("description"),
            "type": "datagen",
            "dataGenerationSpecs": arguments.get("specs"),
        },
        auth=setup_client.client.auth,
    )


def test_list_jobs(setup_client: TestFixture):
    assert setup_client.client.list_jobs() == "the results"
    setup_client.mock_make_request.assert_called_once_with(
        method="get",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/jobs",
        auth=setup_client.client.auth,
    )


def test_describe_job(setup_client: TestFixture):
    assert (
        setup_client.client.describe_job(job_id="some-job")
        == mock_make_request_response
    )
    setup_client.mock_make_request.assert_called_once_with(
        method="get",
        url="some-uri/organizations/mock-org-id/projects/mock-proj-id/jobs/some-job",
        auth=setup_client.client.auth,
    )


@patch(
    "pysolotools.clients.ucvd_client.UCVDClient._UCVDClient__iterable_get",
    autospec=True,
)
def test_iterate_datasets(mock_iterable_get, setup_client: TestFixture):
    mock_iterable_get.return_value = [mock_dataset]
    setup_client.client._UCVDClient__iterable_get = mock_iterable_get
    for result in setup_client.client.iterate_datasets():
        assert isinstance(result, UCVDDataset)
    mock_iterable_get.assert_called_once_with(
        "some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets",
        auth=setup_client.client.auth,
    )


@patch(
    "pysolotools.clients.ucvd_client.UCVDClient._UCVDClient__iterable_get",
    autospec=True,
)
def test_iterate_dataset_archives(mock_iterable_get, setup_client: TestFixture):
    mock_iterable_get.return_value = [mock_dataset_archive]
    setup_client.client._UCVDClient__iterable_get = mock_iterable_get
    for result in setup_client.client.iterate_dataset_archives(
        dataset_id="some-dataset"
    ):
        assert isinstance(result, UCVDArchive)
    mock_iterable_get.assert_called_once_with(
        "some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets/some-dataset/archives",
        auth=setup_client.client.auth,
    )


@patch(
    "pysolotools.clients.ucvd_client.UCVDClient._UCVDClient__iterable_get",
    autospec=True,
)
def test_iterate_dataset_attachments(mock_iterable_get, setup_client: TestFixture):
    mock_iterable_get.return_value = [mock_dataset_attachment]
    setup_client.client._UCVDClient__iterable_get = mock_iterable_get
    for result in setup_client.client.iterate_dataset_attachments(
        dataset_id="some-dataset"
    ):
        assert isinstance(result, UCVDAttachment)
    mock_iterable_get.assert_called_once_with(
        "some-uri/organizations/mock-org-id/projects/mock-proj-id/datasets/some-dataset/attachments",
        auth=setup_client.client.auth,
    )


@pytest.mark.parametrize(
    "arguments,exists,expected_root_path,expected_makedirs",
    [
        ({"solo_dir": "some-dir"}, True, "some-dir", []),
        ({"solo_dir": "some-dir"}, False, "some-dir", [call("some-dir")]),
        (
            {"solo_dir": "some-dir", "subdir": True},
            False,
            "some-dir/solo",
            [call("some-dir/solo")],
        ),
    ],
)
@patch("tarfile.open", autospec=True)
@patch("glob.glob", autospec=True)
@patch("os.path.exists", autospec=True)
@patch("os.makedirs", autospec=True)
def test_extract_dataset(
    mock_makedirs,
    mock_exists,
    mock_glob,
    mock_open,
    setup_client: TestFixture,
    arguments,
    exists,
    expected_root_path,
    expected_makedirs,
):
    mock_exists.return_value = exists
    mock_glob.return_value = ["some-archive"]
    mock_extract = MagicMock()
    mock_open.return_value.__enter__.return_value.extractall = mock_extract
    setup_client.client.extract_dataset(**arguments)

    mock_exists.assert_called_once_with(expected_root_path)
    mock_makedirs.assert_has_calls(expected_makedirs)
    mock_glob.assert_called_once_with("some-dir/*.tar")
    mock_open.assert_called_once_with("some-archive")
    mock_extract.assert_called_once_with(expected_root_path)


@responses.activate
@pytest.mark.parametrize(
    "is_dir,expected_makedirs", [(True, []), (False, [call("some-dir")])]
)
@patch("builtins.open", autospec=True)
@patch.object(UCVDClient, "iterate_dataset_archives", autospec=True)
@patch("os.mkdir", autospec=True)
@patch("os.path.isdir", autospec=True)
def test_download_dataset_archives(
    mock_isdir,
    mock_mkdir,
    mock_iterate_dataset_archives,
    mock_open,
    setup_client: TestFixture,
    is_dir,
    expected_makedirs,
):
    mock_isdir.return_value = is_dir
    mock_iterate_dataset_archives.return_value = [UCVDArchive(**mock_dataset_archive)]
    mock_write = MagicMock()
    mock_open.return_value.__enter__.return_value.write = mock_write
    responses.add(
        responses.GET,
        "https://mock-signed-url",
        stream=True,
        body="stuff",
        status=200,
    )

    setup_client.client.download_dataset_archives(
        dataset_id="some-dataset",
        dest_dir="some-dir",
    )

    mock_open.assert_called_once_with("some-dir/Sample.tar", "wb")
    mock_mkdir.assert_has_calls(expected_makedirs)
    mock_iterate_dataset_archives.assert_called_once_with(
        setup_client.client, "some-dataset"
    )
    mock_write.assert_called_once_with(b"stuff")


@responses.activate
@patch("builtins.open", autospec=True)
@patch.object(UCVDClient, "describe_dataset_attachment", autospec=True)
def test_download_dataset_attachment(
    mock_describe_dataset_attachment, mock_open, setup_client: TestFixture
):
    mock_describe_dataset_attachment.return_value = UCVDAttachment(
        **mock_dataset_attachment
    )
    mock_write = MagicMock()
    mock_open.return_value.__enter__.return_value.write = mock_write
    responses.add(
        responses.GET,
        "https://mock-signed-url",
        stream=True,
        body="stuff",
        status=200,
    )

    setup_client.client.download_dataset_attachment(
        dataset_id="some-dataset",
        attachment_id="some-attachment",
        dest_dir="some-dir",
    )

    mock_open.assert_called_once_with("some-dir/Sample.tar", "wb")
    mock_describe_dataset_attachment.assert_called_once_with(
        setup_client.client, "some-dataset", "some-attachment"
    )
    mock_write.assert_called_once_with(b"stuff")


@patch("requests_toolbelt.StreamingIterator", autospec=True)
@patch("requests.put", autospec=True)
@patch("os.path.getsize", autospec=True)
@patch("builtins.open", autospec=True)
def test_upload_file(
    mock_open, mock_getsize, mock_put, mock_streaming, setup_client: TestFixture
):
    mock_getsize.return_value = 1234
    mock_file_descriptor = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file_descriptor

    assert (
        setup_client.client.upload_file(url="some-url", filename="some-filename")
        == mock_put.return_value
    )

    mock_open.assert_called_once_with("some-filename", "rb")
    mock_streaming.assert_called_once_with(1234, mock_file_descriptor)
    mock_put.assert_called_once_with("some-url", data=mock_streaming.return_value)


@contextmanager
def does_not_raise():
    yield


@responses.activate
@pytest.mark.parametrize(
    "status_code,json_response,expected,expectation",
    [
        (
            200,
            {"next": "1234", "results": ["some stuff"]},
            ["some stuff", "next stuff"],
            does_not_raise(),
        ),
        (200, {"results": ["some stuff"]}, ["some stuff"], does_not_raise()),
        (
            200,
            {"results": ["some stuff", "more stuff"]},
            ["some stuff", "more stuff"],
            does_not_raise(),
        ),
        (404, {}, [], pytest.raises(HTTPError)),
    ],
)
def test___iterable_get(
    setup_client: TestFixture, status_code, json_response, expected, expectation
):
    responses.add(
        responses.GET, url="http://some-url", json=json_response, status=status_code
    )
    responses.add(
        responses.GET, url="http://some-url?next=1234", json={"results": ["next stuff"]}
    )
    items = []
    with expectation:
        for response in setup_client.client._UCVDClient__iterable_get(
            base_url="http://some-url"
        ):
            items.append(response)

    assert items == expected


@pytest.mark.parametrize(
    "raise_for_status,expectation,headers,auth,params,body,data",
    [
        (
            "good",
            does_not_raise(),
            "some headers",
            "some auth",
            "some params",
            "some body",
            "some data",
        ),
        ("good", does_not_raise(), None, None, None, None, None),
        (Exception("bonk"), pytest.raises(UCVDException), None, None, None, None, None),
    ],
)
@patch("requests.Session", autospec=True)
def test___make_request(
    mock_session, raise_for_status, expectation, headers, auth, params, body, data
):
    client = UCVDClient(
        project_id=MOCK_PROJECT_ID,
        org_id=MOCK_ORG_ID,
        sa_key=MOCK_SA_KEY,
        api_secret=MOCK_API_SECRET,
        base_uri=BASE_URI,
    )

    mock_request = create_autospec(Request)
    mock_response = create_autospec(Response)
    mock_response.json.return_value = {"some": "stuff"}
    mock_response.raise_for_status = Mock(side_effect=raise_for_status)
    mock_request.return_value = mock_response
    mock_session.return_value.request = mock_request

    with expectation:
        result = client._UCVDClient__make_request(
            method="some-method",
            url="some-url",
            headers=headers,
            auth=auth,
            params=params,
            body=body,
            data=data,
        )
        assert result == {"some": "stuff"}

    mock_session.assert_called_once()
    mock_request.assert_called_once_with(
        method="some-method",
        url="some-url",
        headers=headers,
        auth=auth,
        params=params or {},
        json=body,
        data=data,
    )
    mock_session.return_value.close.assert_called_once()
