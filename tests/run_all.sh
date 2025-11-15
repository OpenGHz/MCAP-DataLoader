#!/bin/bash

# 默认使用脚本所在目录
DEFAULT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 解析命令行参数
if [ $# -eq 0 ]; then
    TARGET_DIR="$DEFAULT_DIR"
    echo "未指定目录，使用脚本所在目录: $TARGET_DIR"
elif [ $# -eq 1 ]; then
    TARGET_DIR="$1"
    if [ ! -d "$TARGET_DIR" ]; then
        echo "错误：'$TARGET_DIR' 不是一个有效目录。" >&2
        exit 1
    fi
    echo "使用指定目录: $TARGET_DIR"
else
    echo "用法: $0 [目录路径]" >&2
    echo "      若不指定目录，则默认执行脚本所在目录下的所有 .py 文件。" >&2
    exit 1
fi

echo "----------------------------------------"

error_occurred=false

# 查找目标目录下所有 .py 文件（仅当前层，不递归），按字典序排序
while IFS= read -r -d '' pyfile; do
    filename=$(basename "$pyfile")
    echo "▶ 执行: $filename"
    
    if python3 "$pyfile"; then
        echo "✅ 成功: $filename"
    else
        echo "❌ 失败: $filename （已记录错误，继续执行下一个）"
        error_occurred=true
    fi
    echo ""
done < <(find "$TARGET_DIR" -maxdepth 1 -type f -name "*.py" -print0 | sort -z)

echo "----------------------------------------"
if [ "$error_occurred" = true ]; then
    echo "⚠️  注意：部分脚本执行失败，请检查上述错误。"
    exit 1
else
    echo "🎉 所有 Python 脚本均执行成功！"
    exit 0
fi