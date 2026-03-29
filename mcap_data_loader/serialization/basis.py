from mcap.reader import make_reader
from mcap.records import Schema, Message
from turbojpeg import TurboJPEG
from typing import Dict, IO, Set, Optional, List, Any, Union, final
from collections.abc import Generator, Iterable
from functools import cache, cached_property
from mcap_data_loader.basis import DictDataStamped
from mcap_data_loader.utils.basic import zip
from mcap_data_loader.utils.av_coder import AvCoder, DecodeConfig
from mcap_data_loader.utils.stat import StatisticsBasis, Statistics
from collections import defaultdict
from abc import ABC, abstractmethod
import json
import numpy as np
import logging


CONFIG_TYPE_TO_MEDIA_TYPE = {DecodeConfig: "video/mp4"}


class McapReaderBasis(ABC):
    decoder_factories = ()

    def __init__(self, file: IO[bytes]):
        self.file_io = file
        self._file_path = getattr(file, "name", None)
        self.reader = make_reader(file, False, self.decoder_factories)
        self._stat_schemas = ()
        self._post_init()
        self._jpeg = None

    def _post_init(self):
        return

    def iter_message_samples(
        self,
        topics: Optional[Iterable[str]] = None,
        reverse: bool = False,
    ) -> Generator[DictDataStamped]:
        """Iterate over messages in the MCAP file."""
        # TODO: support iter through a reference topic
        # and inter other topics with start_time according
        # to the reference topic
        if topics is None:
            topics = self.all_topic_names()
        else:
            diff = set(topics) - self.all_topic_names()
            assert not diff, (
                f"Topics {diff} not found. Available: {self.all_topic_names()}"
            )
        messages = {}
        for schema, channel, message in self.reader.iter_messages(
            topics, reverse=reverse
        ):
            messages[channel.topic] = {
                "data": self._decode(schema, message),
                "t": message.publish_time,
            }
            if len(messages) == len(topics):
                yield messages
                messages.clear()

    @abstractmethod
    def _decode(self, schema: Schema, message: Message) -> Any:
        pass

    @cache
    def all_topic_names(self) -> Set[str]:
        """Get all topics in the MCAP file."""
        return {
            channel.topic for channel in self.reader.get_summary().channels.values()
        }

    @cache
    def all_attachment_names(self) -> Set[str]:
        """Get all attachment names in the MCAP file."""
        return {attachment.name for attachment in self.reader.iter_attachments()}

    def iter_attachment_samples(
        self,
        names: Optional[Iterable[str]] = None,
        reverse: bool = False,
        configs: Optional[list] = None,
    ) -> Generator[Union[DictDataStamped, Any]]:
        """Iterate over target attachments in the MCAP file."""
        assert not reverse, "Reverse iteration is not supported for attachments yet."
        media_config = (
            {CONFIG_TYPE_TO_MEDIA_TYPE[type(config)]: config for config in configs}
            if configs
            else {}
        )
        if names is None:
            names = self.all_attachment_names()
        else:
            names = set(names)
            diff = set(names) - self.all_attachment_names()
            assert not diff, (
                f"Attachments {diff} not found. Available: {self.all_attachment_names()}"
            )

        attch_names: List[str] = []
        iters: List[Iterable] = []
        for attachment in self.reader.iter_attachments():
            name = attachment.name
            if name in names:
                media_type = attachment.media_type
                cfg = media_config.get(media_type, None)
                if media_type == "video/mp4":
                    coder = AvCoder()
                    # FIXME: check whether now no mismatching
                    attach_iter = coder.iter_decode(attachment.data, cfg)
                elif media_type == "application/json":
                    attach_iter = json.loads(attachment.data)
                else:
                    raise ValueError(f"Unsupported media type: {media_type}")
                attch_names.append(name)
                iters.append(attach_iter)
                if len(attch_names) == len(names):
                    break
        else:
            assert not names, (
                f"Not all requested attachments found: {names} vs {attch_names}"
            )
        try:
            for values in zip(*iters):
                data = {}
                for name, value in zip(attch_names, values):
                    data[name] = value
                yield data
        except ValueError as e:
            raise ValueError(
                f"Attachment iterators have different lengths: {attch_names}"
            ) from e

    def iter_samples(
        self,
        keys: Optional[Iterable[str]] = (),
        topics: Optional[Iterable[str]] = (),
        attachments: Optional[Iterable[str]] = (),
        reverse: bool = False,
        strict: bool = True,
        with_step: bool = False,
        with_file: bool = False,
        extra_keys: bool = False,
        configs: Optional[list] = None,
    ) -> Generator[DictDataStamped[np.ndarray]]:
        """Iterate over messages and attachments in the MCAP file.
        Args:
            keys (Optional[Iterable[str]]): Specific keys to include in the samples.
                The keys can be topic names or attachment names. If None, will ignore this filter.
                If provided, the keys must be unique across topics and attachments.
            topics (Optional[Iterable[str]]): Specific topics to include in the samples.
                If None, will include all topics.
            attachments (Optional[Iterable[str]]): Specific attachments to include in the samples.
                If None, will include all attachments.
            reverse (bool): Whether to iterate in reverse order.
            strict (bool): Whether to enforce strict length matching between topic and attachment iterators.
            with_step (bool): Whether to include the step information in the yielded data.
            with_file (bool): Whether to include the file path in the yielded data.
        Returns:
            Generator[Dict[str, Any]]: A generator yielding dictionaries containing message and attachment data.
        Raises:
            ValueError: If the keys are not unique across topics and attachments.
            ValueError: If the topics or attachments are not found.
        """
        all_topics = self.all_topic_names()
        all_attachments = self.all_attachment_names()
        topics = set(topics) if topics is not None else all_topics
        attachments = set(attachments) if attachments is not None else all_attachments
        keys = keys or []
        for key in keys:
            flag = 0
            if key in all_topics:
                topics.add(key)
                flag += 1
            if key in all_attachments:
                attachments.add(key)
                flag += 1
            if flag == 0 and not extra_keys:
                raise ValueError(
                    f"Key '{key}' not found in topics or attachments. Available topics: {all_topics}, attachments: {all_attachments}."
                )
            elif flag > 1:
                raise ValueError(
                    f"Key '{key}' found in both topics and attachments, please specify only one."
                )

        if extra_keys:
            removed = set()
            for topic in topics.copy():
                if topic not in all_topics:
                    topics.remove(topic)
                    removed.add(topic)
            for attachment in attachments.copy():
                if attachment not in all_attachments:
                    attachments.remove(attachment)
                    removed.add(attachment)
            if removed:
                self.get_logger().info(f"Keys {removed} not found and will be ignored.")

        def empty_iter():
            for _ in range(len(self)):
                yield {}

        # The first iteration costs more time since it needs to create these iterators.
        topic_iter = (
            self.iter_message_samples(topics, reverse) if topics else empty_iter()
        )
        attachment_iter = (
            self.iter_attachment_samples(attachments, reverse, configs)
            if attachments
            else empty_iter()
        )
        file_path = np.array(self._file_path)
        try:
            for step, (msg_data, att_data) in enumerate(
                zip(topic_iter, attachment_iter)
            ):
                data = msg_data | att_data
                if with_step:
                    data["step"] = {"t": 0, "data": np.array(step)}
                if with_file:
                    data["file"] = {"t": 0, "data": file_path}
                yield data
        except ValueError as e:
            error = "Topic and attachment iterators have different lengths"
            if strict:
                raise ValueError(error) from e
            self.get_logger().warning(error)

    @cache
    def topic_message_counts(self) -> Dict[str, int]:
        """Get the message count for each topic in the MCAP file."""
        topic_msg_count = {}
        summary = self.reader.get_summary()
        statistics = summary.statistics
        for c_id, stats in statistics.channel_message_counts.items():
            # get topic name from channel id
            topic = summary.channels[c_id].topic
            topic_msg_count[topic] = stats
        return topic_msg_count

    @staticmethod
    def equal_message_counts(counts: Dict[str, int]) -> int:
        """Check if all topics have the same number of messages.
        Args:
            counts (Dict[str, int]): A dictionary mapping topic names to their message counts.
        Returns:
            int: The common message count if all topics have the same count, otherwise 0.
        Raises:
            AssertionError: If the counts dictionary is empty or contains non-positive counts.
        """
        assert counts, "Counts dictionary is empty"
        counts = list(counts.values())
        first_count = counts[0]
        for count in counts[1:]:
            assert count > 0, "Message count must be positive"
            if count != first_count:
                return 0
        return first_count

    @final
    def close(self):
        """Close the MCAP file."""
        if not self.file_io.closed:
            self.file_io.close()

    @classmethod
    def get_logger(cls) -> logging.Logger:
        return logging.getLogger(cls.__name__)

    def has_topic_statistics(self) -> bool:
        """Check if the MCAP file has topic statistics attachment."""
        return "topic_statistics" in self.all_attachment_names()

    def _process_stats(
        self, stats: Dict[str, StatisticsBasis]
    ) -> Dict[str, Statistics]:
        for topic, stat in stats.items():
            cnt = self.topic_message_counts()[topic]
            stat["mean"] = stat["sum"] / cnt
            var = stat["sum_sq"] / cnt - stat["mean"] ** 2
            stat["std"] = np.maximum(var, 0.0) ** 0.5
            stat["n"] = cnt
        return stats

    def compute_topic_statistics(self) -> Dict[str, Statistics]:
        """Compute statistics for each topic in the MCAP file."""
        if self.reader is None:
            raise ValueError("Reader is not initialized.")

        stat_topics = set()
        for schema, channel, message in self.reader.iter_messages():
            if schema.name in self._stat_schemas:
                stat_topics.add(channel.topic)
        stats = defaultdict(
            lambda: {"sum": 0, "sum_sq": 0, "min": float("inf"), "max": float("-inf")}
        )
        for sample in self.iter_message_samples(stat_topics):
            for topic in stat_topics:
                data = sample[topic]["data"]
                stats[topic]["sum"] += data
                stats[topic]["sum_sq"] += data**2
                stats[topic]["min"] = np.minimum(stats[topic]["min"], data)
                stats[topic]["max"] = np.maximum(stats[topic]["max"], data)
        return self._process_stats(stats)

    @cached_property
    def topic_statistics(self) -> Dict[str, Statistics]:
        """Get the topic statistics attachment from the MCAP file."""
        for attach in self.reader.iter_attachments():
            if attach.name == "topic_statistics":
                stats: Dict[str, StatisticsBasis] = json.loads(attach.data)
                for stat in stats.values():
                    for name, value in stat.items():
                        stat[name] = np.asarray(value)
                return self._process_stats(stats)
        self.get_logger().info("Computing topic statistics...")
        return self.compute_topic_statistics()

    @cache
    def __len__(self) -> int:
        """Get the total number of messages in the MCAP file."""
        counts = self.topic_message_counts()
        length = self.equal_message_counts(counts)
        if length == 0:
            if counts:
                raise ValueError(
                    f"Not all topics have the same number of messages. Counts: {counts}"
                )
            else:
                raise ValueError("No messages found in the MCAP file.")
        return length

    def __del__(self):
        self.close()

    @property
    def jpeg(self) -> Optional[TurboJPEG]:
        """Get the TurboJPEG decoder instance, or None if it failed to initialize."""
        if self._jpeg is None:
            self._jpeg = TurboJPEG()
        return self._jpeg
