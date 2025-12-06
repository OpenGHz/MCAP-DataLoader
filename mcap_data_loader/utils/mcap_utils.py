from pymcap import PyMCAP
from pymcap.core import McapCLIOutput
from mcap.writer import Writer
from mcap.reader import McapReader, make_reader
from typing import Optional, List, Dict, Union, Literal, Any
from pathlib import Path
from time import time_ns
from mcap_data_loader.utils.basic import StrEnum
import shutil
import uuid
import json
import tempfile
import logging


class McapCLI(PyMCAP):
    """Class to interact with the MCAP command-line interface (CLI) tool."""

    def _PyMCAP__get_executable(self):
        executable_path = shutil.which("mcap")
        if executable_path is None:
            return super().__get_executable()
        else:
            return executable_path

    def add_metadata(self, file_path: str, name: str, metadata: Dict[str, Any]):
        """Add metadata to the MCAP file."""
        pair_str = " ".join([f"{k}={json.dumps(v)}" for k, v in metadata.items()])
        return self.check_cmd_output(
            self.run_command(f"add metadata {file_path} -n {name} -k {pair_str}")
        )

    def add_attachment(
        self,
        file_path: str,
        create_time: int,
        log_time: int,
        name: str,
        media_type: str,
        data: Union[str, bytes],
    ):
        """Add attachment to the MCAP file.
        Args:
            file_path (str): Path to the MCAP file.
            create_time (int): Creation time in nanoseconds.
            log_time (int): Log time in nanoseconds.
            name (str): Name of the attachment.
            media_type (str): Media type of the attachment. See MediaType enum.
            data (str | bytes): Data to be added as attachment. If str, it is treated as file path.
        Returns:
            str: Standard output from the MCAP CLI command.
        """
        use_tmp_file = False
        if isinstance(data, bytes):
            use_tmp_file = True
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(data)
                tmp_file.flush()
                file = tmp_file.name
        else:
            file = data
        output = self.run_command(
            f"add attachment {file_path} -n {name} --content-type {media_type} "
            # x-time has bugs in current mcap CLI
            # f"--log-time {log_time} --creation-time {create_time} "
            f"-f {file}"
        )
        if use_tmp_file:
            tmp_file.close()
            Path(tmp_file.name).unlink()
        return self.check_cmd_output(output)

    def check_cmd_output(self, output: McapCLIOutput):
        """Check the output of an MCAP CLI command."""
        if not output.success:
            raise RuntimeError(
                f"MCAP command failed with error: {output.stderr.strip()}"
            )
        return output.stdout.strip()


DeriveMetadata = Dict[Literal["self", "parents"], Dict[str, Any]]


class McapHandlerBasis:
    def __init__(
        self, writer: Optional[Writer] = None, reader: Optional[McapReader] = None
    ):
        self.writer = writer
        self.reader = reader

    def set_writer(self, writer: Writer, replace: bool = False):
        """Set the MCAP writer."""
        if self.writer is not None and not replace:
            raise ValueError("Writer is already set. Use `replace=True` to replace it.")
        self.writer = writer

    def set_reader(self, reader: McapReader, replace: bool = False):
        """Set the MCAP reader."""
        if self.reader is not None and not replace:
            raise ValueError("Reader is already set. Use `replace=True` to replace it.")
        self.reader = reader


class McapDeriveMetadataHandler(McapHandlerBasis):
    """Handler for MCAP derivation metadata."""

    def write(
        self,
        self_uuid: str = "",
        configs: Optional[List[Dict[str, Any]]] = None,
        parent_derive_info: Optional[Dict[str, Dict[str, Any]]] = None,
        parent_file_paths: Optional[List[str]] = None,
    ):
        """Add derivation metadata to the MCAP file.
        The derivation means that this file is derived (computed) from other files, which
        can be used to ensure reproducibility, traceability and consistency.
        Args:
            self_uuid (str): UUID of the current file. If empty, a new UUID will be generated.
            parent_derive_info (Dict[str, Dict]): A dictionary mapping parent file paths to their derivation metadata.
            parent_file_paths (List[str]): List of parent file paths. If None, use keys of parent_derive_info.
        Raises:
            ValueError: If the writer is not set or if the parent file's derive info is
                        missing or does not match the given info.
        """
        self.derive_configs = configs or []
        self.parent_derive_info = parent_derive_info or {}
        self.parent_file_paths = parent_file_paths or self.parent_derive_info.keys()
        parent_uuids = []
        for parent_file_path in self.parent_file_paths:
            # Read parent file derive info
            with open(parent_file_path, "rb") as f:
                reader = make_reader(f)
                parent_info = self._read(reader)
            given_info = self.parent_derive_info.get(parent_file_path, {})
            write_parent = False
            if parent_info:
                if parent_info != given_info:
                    raise ValueError(
                        f"Parent file {parent_file_path} derive info does not match the given info:"
                        f" {parent_info} != {given_info}"
                    )
            elif not given_info:
                raise ValueError(
                    f"Parent file {parent_file_path} derive info is missing. Please provide it."
                )
            else:
                parent_info = given_info
                write_parent = True

            parent_uuid = parent_info.get("self", {}).get("uuid", "")
            if not parent_uuid:
                raise ValueError(
                    f"Parent file {parent_file_path} does not have a valid 'self' UUID."
                )
            if write_parent:
                self.get_logger().info(
                    f"Writing given derive info for parent file {parent_file_path}"
                )
                McapCLI().add_metadata(parent_file_path, "derive", parent_info)
            parent_uuids.append(parent_uuid)

        self.writer.add_attachment(
            time_ns(),
            time_ns(),
            "derive",
            MediaType.APPLICATION_JSON,
            json.dumps(
                {
                    "self": {
                        "uuid": self_uuid or uuid.uuid1().hex,
                        "configs": self.derive_configs,
                    },
                    "parents": parent_uuids,
                }
            ),
        )

    @staticmethod
    def _read(reader: McapReader) -> DeriveMetadata:
        """Get derivation metadata from the MCAP file.
        Returns:
            dict: A dictionary containing 'self' and 'parents' UUIDs.
        """
        derive = {}
        for data in reader.iter_attachments():
            if data.name == "derive":
                return json.loads(data.data)
        return derive

    def read(self) -> DeriveMetadata:
        if self.reader is None:
            raise ValueError("Reader is not initialized.")
        return self._read(self.reader)

    def finish(self):
        """Finish writing or close reading the MCAP file."""
        if self.writer:
            self.writer.finish()
        # reader does not have a finish method

    @classmethod
    def get_logger(cls) -> logging.Logger:
        return logging.getLogger(cls.__name__)


class MediaType(StrEnum):
    TEXT_PLAIN = "text/plain"
    TEXT_HTML = "text/html"
    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"
    VIDEO_MP4 = "video/mp4"
    APPLICATION_JSON = "application/json"
    APPLICATION_OCTET_STREAM = "application/octet-stream"


class ArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


class McapTool(McapHandlerBasis):
    """Utility class for MCAP file operations."""

    def add_log_stamps_attachment(self, log_stamps: List[int]):
        self.writer.add_attachment(
            time_ns(),
            time_ns(),
            "log_stamps",
            MediaType.APPLICATION_JSON,
            json.dumps(log_stamps).encode(),
        )

    @staticmethod
    def topic_statistics_attachment_args(statistics: dict):
        return (
            time_ns(),
            time_ns(),
            "topic_statistics",
            MediaType.APPLICATION_JSON,
            json.dumps(statistics, cls=ArrayEncoder).encode(),
        )

    def add_topic_statistics_attachment(self, statistics: dict):
        self.writer.add_attachment(*self.topic_statistics_attachment_args(statistics))
