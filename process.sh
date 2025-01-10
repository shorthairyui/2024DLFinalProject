#!/bin/bash

# 接收输入文件、输出文件路径和关键词
input_file=$1
output_file=$2
keywords=$3

# 检查输入文件是否存在
if [[ ! -f "$input_file" ]]; then
  echo "输入文件不存在"
  exit 1
fi
echo "123"
# 调用 Python 脚本处理文件，传递关键词
python3 process_image.py "$input_file" "$output_file" "$keywords"
exit $?
