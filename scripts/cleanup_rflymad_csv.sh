#!/bin/bash
# 清理 RflyMAD 中 ULog 转出的 CSV，保留原始 ULog 和必要的 9 个 CSV
# 释放空间以继续处理

set -e

KEEP_PATTERNS="actuator_armed\|actuator_outputs\|estimator_attitude\|estimator_status\|sensor_combined\|vehicle_command\|vehicle_global_position\|vehicle_gps_position\|vehicle_magnetometer"

echo "开始清理 RflyMAD 中生成的 CSV..."
echo ""

# 统计要删除的文件
TO_DELETE=$(find /home/jack/RflyMAD -path "*/Log/*.csv" -type f | grep -v "$KEEP_PATTERNS" | wc -l)
echo "将删除 $TO_DELETE 个 CSV 文件"

# 计算大小
SIZE=$(find /home/jack/RflyMAD -path "*/Log/*.csv" -type f | grep -v "$KEEP_PATTERNS" | xargs du -ch 2>/dev/null | tail -1 | cut -f1)
echo "预计释放空间: $SIZE"
echo ""

read -p "确认删除？(y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "删除中..."
    find /home/jack/RflyMAD -path "*/Log/*.csv" -type f | grep -v "$KEEP_PATTERNS" | xargs rm -f
    echo "完成！"
    echo ""
    df -h /home/jack --output=avail,pcent | tail -1
else
    echo "已取消"
fi
