#!/bin/bash
# 批量将 RflyMAD 中所有 ULog 文件转换为完整 CSV
# 只处理缺少 vehicle_land_detected CSV 的目录

source /home/jack/miniconda3/bin/activate UAV-Fault-Diagnosis

total=0
converted=0
skipped=0
failed=0

# 遍历所有 ULog 文件
while IFS= read -r ulg; do
    dir=$(dirname "$ulg")
    ((total++))

    # 检查是否已有 vehicle_land_detected（说明已转换过）
    if ls "$dir"/*vehicle_land_detected* >/dev/null 2>&1; then
        ((skipped++))
        continue
    fi

    # 转换
    cd "$dir"
    if ulog2csv "$(basename "$ulg")" >/dev/null 2>&1; then
        ((converted++))
    else
        ((failed++))
    fi

    # 每 100 个打印进度
    if [ $((total % 100)) -eq 0 ]; then
        echo "进度: $total / 6605 (转换: $converted, 跳过: $skipped, 失败: $failed)"
    fi

done < <(find /home/jack/RflyMAD -name "*.ulg" -type f | sort)

echo ""
echo "========================================"
echo "ULog 转换完成"
echo "总计: $total, 新转换: $converted, 已存在: $skipped, 失败: $failed"
echo "========================================"
