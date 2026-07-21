import argparse
import json
import subprocess
import sys
import yaml
from typing import Any, Dict


# Dict keys whose value is kept as ONE whole mapping (JSON-encoded) instead of
# being flattened into --a.b.c=... args. There are NO built-in defaults: a config
# declares exactly which keys to preserve via a top-level ``preserve_keys:`` list
# in its YAML (see extract_preserve_keys / get_flat_args_dict), so behavior is
# explicit per-config and adding a new structured arg needs no code edit.

# Reserved top-level YAML key holding the preserve keys for this run. It is
# consumed (popped) before flattening and never emitted as a downstream arg.
PRESERVE_KEYS_FIELD = "preserve_keys"


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
    preserve_keys: Any = None,
) -> Dict[str, Any]:
    """
    递归地将嵌套字典扁平化，例如：
    {'a': {'b': 1}} -> {'a.b': 1}

    ``preserve_keys`` 为需要整体保留(不向下拍平)的键集合，来自配置里的
    ``preserve_keys:`` 声明；命中的 dict 值原样保留，便于把结构化参数(如自定义
    相机映射)整体传给下游。无内置默认，全部由配置显式声明。
    """
    preserve = set(preserve_keys) if preserve_keys else set()
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and k not in preserve:
            items.extend(
                flatten_dict(v, new_key, sep=sep, preserve_keys=preserve).items()
            )
        else:
            items.append((new_key, v))
    return dict(items)


def extract_preserve_keys(config: Dict[str, Any]) -> list:
    """Pop the top-level ``preserve_keys`` list from a loaded config (if any).

    Returns the extra preserve keys and mutates ``config`` to remove the field so
    it is not flattened/emitted downstream.
    """
    value = config.pop(PRESERVE_KEYS_FIELD, None)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def value_to_str(value: Any) -> str:
    """将 YAML 中的值安全地转为字符串（保留布尔、数字等原意）"""
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (dict, list)):
        # 用 JSON 保持结构化参数稳定，便于下游解析
        return json.dumps(value, ensure_ascii=True)
    elif value is None:
        return "null"
    else:
        return str(value)


def parse_args(exclude=None, add_help=True, add_command_args=True):
    parser = argparse.ArgumentParser(
        description="Run a command with arguments from a YAML config file, "
        "converted to --key.subkey=value style.",
        add_help=add_help,
    )
    if add_command_args:
        parser.add_argument(
            "command",
            nargs="+",
            help="The command to run (e.g., python train.py)",
        )
    parser.add_argument(
        "--config", "-c", required=True, help="Path to the YAML config file"
    )
    parser.add_argument(
        "--exclude",
        "-e",
        nargs="*",
        default=exclude or [],
        help="Top-level keys to exclude from the config",
    )
    if not add_help and "-h" in sys.argv:
        return None
    return parser.parse_known_args()[0]


def get_flat_args_dict(args) -> Dict[str, Any]:
    try:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading config file {args.config}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(config, dict):
        print(
            f"Config file {args.config} must contain a YAML mapping (dictionary).",
            file=sys.stderr,
        )
        sys.exit(1)
    for field in args.exclude:
        config.pop(field, None)
    preserve_keys = extract_preserve_keys(config)
    flat_config = flatten_dict(config, preserve_keys=preserve_keys)
    return flat_config


def get_args_list(args) -> list:
    flat_config = get_flat_args_dict(args)
    extra_args = []
    for key, value in flat_config.items():
        arg_str = f"--{key}={value_to_str(value)}"
        extra_args.append(arg_str)
    return extra_args


def main_func(args):
    extra_args = get_args_list(args)
    # 完整命令 = 用户命令 + 额外参数
    full_cmd = args.command + extra_args

    print("Executing:", " ".join(full_cmd), file=sys.stderr)

    # 执行命令
    try:
        result = subprocess.run(full_cmd)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"Command not found: {args.command[0]}", file=sys.stderr)
        sys.exit(127)

    return 0


def main(exclude=None) -> int:
    return main_func(parse_args(exclude=exclude))


if __name__ == "__main__":
    import sys

    sys.exit(main())
