import sys


import sys


def extract_and_remove_args(target_args, argv=None, inplace=True):
    """
    从 argv 中提取并移除指定的参数（及其值），返回剩余参数和提取出的参数字典。

    支持两种参数形式：
      - 带值参数：-c config.yaml → {'-c': 'config.yaml'}
      - 无值参数（flag）：--verbose → {'--verbose': None}

    参数:
        target_args (list of str): 要提取的参数名列表，如 ['-c', '--config', '--verbose']。
        argv (list of str, optional): 原始命令行参数列表，默认为 sys.argv。
        inplace (bool): 是否原地修改传入的 argv 列表。

    返回:
        tuple: (remaining_argv, extracted_dict)
            - remaining_argv: 移除了目标参数后的新参数列表
            - extracted_dict: {arg_name: arg_value or None}
    """
    argv = argv if argv is not None else sys.argv
    target_set = set(target_args)
    remaining = []
    extracted = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in target_set:
            # 尝试获取下一个参数作为值
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                # 下一个存在且不是选项 → 视为值
                extracted[arg] = argv[i + 1]
                i += 2  # 跳过参数和值
            else:
                # 没有下一个，或下一个是选项 → 当作无值 flag
                extracted[arg] = None
                i += 1  # 只跳过当前参数
        else:
            remaining.append(arg)
            i += 1

    if inplace:
        argv.clear()
        argv.extend(remaining)

    return remaining, extracted


def extend_args(base_args: list, extra_args: list):
    """
    将 extra_args 中的参数添加到 base_args 中。
    如果 base_args 中已存在相同的参数键（'='前的部分），则跳过，不覆盖。

    参数:
        base_args (list of str): 原始参数列表，例如 sys.argv。
        extra_args (list of str): 需要添加的参数列表，如 ['--key=value', '--flag']。
    返回:
        list of str: 更新后的参数列表。
    """

    def extract_key(arg: str) -> str:
        """提取参数的 key 部分（等号前的内容，若无等号则返回原字符串）"""
        if "=" in arg:
            return arg.split("=", 1)[0]
        else:
            return arg

    # 构建已存在的 key 集合（只看 key，忽略值）
    existing_keys = set(extract_key(arg) for arg in base_args)

    # 遍历 extra_args，只添加 key 未出现过的参数
    for arg in extra_args:
        key = extract_key(arg)
        if key not in existing_keys:
            base_args.append(arg)
            existing_keys.add(key)  # 保持集合同步

    return base_args


def extend_and_override_args(base_args, extra_args):
    """
    将 extra_args 中的参数覆盖或添加到 base_args 中。

    参数:
        base_args (list of str): 原始参数列表，例如 sys.argv。
        extra_args (dict): 需要覆盖或添加的参数字典，如 {'--key': 'value'}。
    返回:
        list of str: 更新后的参数列表。
    """
    # 首先将 base_args 转换为字典以便覆盖
    arg_dict = {}
    i = 0
    while i < len(base_args):
        arg = base_args[i]
        if (
            arg.startswith("-")
            and i + 1 < len(base_args)
            and not base_args[i + 1].startswith("-")
        ):
            arg_dict[arg] = base_args[i + 1]
            i += 2
        else:
            # 保留非选项参数（如命令或位置参数）
            arg_dict[arg] = None
            i += 1

    # 更新字典
    arg_dict.update(extra_args)

    # 将字典转换回列表
    new_args = []
    for key, value in arg_dict.items():
        new_args.append(key)
        if value is not None:
            new_args.append(value)

    return new_args


if __name__ == "__main__":
    # 示例输入
    argv = [
        "script.py",
        "-i",
        "input.txt",
        "-c",
        "config.yaml",
        "--verbose",
        "-o",
        "out.txt",
    ]
    targets = ["-c", "--config", "-o", "--output", "--verbose"]

    new_argv, config = extract_and_remove_args(targets, argv)

    print("剩余参数:", new_argv)
    # 输出: ['script.py', '-i', 'input.txt']

    print("提取的配置:", config)
    # 输出: {'-c': 'config.yaml', '-o': 'out.txt', '--verbose': None}

    # 示例覆盖
    extra = {"--verbose": "true", "-o": "final_output.txt"}
    updated_argv = extend_and_override_args(new_argv, extra)
    print("更新后的参数:", updated_argv)
    # 输出: ['script.py', '-i', 'input.txt', '--verbose', 'true', '-o', 'final_output.txt']

    # 示例添加
    extra_add = {"--new": "value", "--verbose": "false"}
    updated_argv_add = extend_args(new_argv, extra_add)
    print("添加后的参数:", updated_argv_add)
    # 输出: ['script.py', '-i', 'input.txt', '--verbose', '--new', 'value']
