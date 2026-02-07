#!/usr/bin/env python3
import argparse
import subprocess
import sys
import yaml
from typing import Any, Dict


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    递归地将嵌套字典扁平化，例如：
    {'a': {'b': 1}} -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def value_to_str(value: Any) -> str:
    """将 YAML 中的值安全地转为字符串（保留布尔、数字等原意）"""
    if isinstance(value, bool):
        # 避免变成小写 true/false（某些命令可能需要 True/False 或 1/0）
        # 这里按字符串输出，由目标命令解释
        return str(value).lower()
    elif value is None:
        return "null"
    else:
        return str(value)


def main(exclude=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a command with arguments from a YAML config file, "
        "converted to --key.subkey=value style."
    )
    parser.add_argument(
        "command", nargs="+", help="The command to run (e.g., python train.py)"
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

    args = parser.parse_args()

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
    # 扁平化配置
    flat_config = flatten_dict(config)

    # 构造 --key=value 参数列表
    extra_args = []
    for key, value in flat_config.items():
        arg_str = f"--{key}={value_to_str(value)}"
        extra_args.append(arg_str)

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


if __name__ == "__main__":
    import sys

    sys.exit(main())
