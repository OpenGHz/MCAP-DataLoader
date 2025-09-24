#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HoloViews 流式多曲线图可视化（每个 field、每个 index 独立面板）

变更点（满足用户需求）：
- 不同 field 作为不同布局维度；同一 field 下不同 index 也作为独立面板。
- 使用 HoloViews Buffer 为每个 (field, index) 创建独立曲线，所有曲线随时间同步滚动。
- 使用 NdLayout(kdims=["field", "index"]) 组织为网格，便于同时查看。

运行环境提示：未安装 Panel/HoloViews 时会给出友好提示。

手动范围设置：
- 可通过设置模块变量 `MANUAL_YLIM` 固定 y 轴范围。
- 赋值为元组 `(ymin, ymax)` 则对所有子图生效；
- 或赋值为字典 `{field_name: (ymin, ymax)}` 仅对指定 field 生效，其余仍按自动范围。
示例：
    MANUAL_YLIM = (-20.0, 40.0)
    # 或仅固定 Field_1/Field_2：
    MANUAL_YLIM = {"Field_1": (-5, 25), "Field_2": (-10, 12)}
"""

import numpy as np
import time
import threading
from typing import Dict, Any, Optional, Union, Tuple


# 注意：避免在导入阶段进行 hv/pn 的初始化，将其延后到 main() 中。

# 手动设定 y 轴范围：
# - 设为 None 表示不强制手动范围（默认，使用自动/半自动方式）。
# - 设为 (ymin, ymax) 则所有子图统一固定该范围。
# - 设为 {field_name: (ymin, ymax)} 则对指定 field 固定范围。
MANUAL_YLIM: Optional[Union[Tuple[float, float], Dict[str, Tuple[float, float]]]] = None


# ----------------------------
# 2. 生成模拟数据
# ----------------------------
def generate_sample_data(num_timesteps=100, num_fields=4, samples_per_field=5):
    """
    生成模拟时序数据
    :param num_timesteps: 时间步数量
    :param num_fields: 字段数量
    :param samples_per_field: 每个字段的样本数（即value列表长度）
    :return: 长度为num_timesteps的列表，每个元素是字典 {field_name: [float, float, ...]}
    """
    np.random.seed(42)  # 固定随机种子以便复现
    field_names = [f"Field_{i + 1}" for i in range(num_fields)]
    data = []

    for t in range(num_timesteps):
        data_at_t = {}
        for field in field_names:
            # 生成一些有趋势的模拟数据，使其可视化效果更明显
            base_value = t * 0.1  # 随时间缓慢上升的基线
            noise = np.random.randn(samples_per_field) * 2  # 添加噪声
            if field == "Field_1":
                values = base_value + noise + 10
            elif field == "Field_2":
                values = base_value + noise + 5 * np.sin(t * 0.2)
            elif field == "Field_3":
                values = base_value + noise + 20
            else:  # Field_4
                values = base_value + noise - 5
            data_at_t[field] = values.tolist()
        data.append(data_at_t)

    return data


def main():
    try:
        import holoviews as hv
        from holoviews import opts
        from holoviews.streams import Buffer
        import panel as pn
    except Exception as e:
        print("未找到 Panel/HoloViews 依赖或初始化失败：", e)
        print("请安装依赖后重试，例如：pip install panel holoviews bokeh")
        return

    # 初始化可视化后端（仅在运行时执行）
    hv.extension("bokeh")
    pn.extension()

    # 生成数据并设置共享状态
    time_series_data = generate_sample_data(
        num_timesteps=3, num_fields=4, samples_per_field=5
    )
    total_steps = len(time_series_data)
    current_step = [0]

    # 推断维度信息
    first_sample: Dict[str, Any] = time_series_data[0]
    field_names = list(first_sample.keys())
    samples_per_field = len(next(iter(first_sample.values())))

    # 为每个 field 计算全局 y 轴范围（带 10% 余量），以避免刷新时跳变
    field_ylim: Dict[str, tuple[float, float]] = {}
    for field in field_names:
        vals = []
        for step in range(total_steps):
            vals.extend(time_series_data[step][field])
        if len(vals) == 0:
            ymin, ymax = -1.0, 1.0
        else:
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            span = vmax - vmin
            if span == 0:
                span = 1.0
            pad = span * 0.1
            ymin, ymax = vmin - pad, vmax + pad
        field_ylim[field] = (ymin, ymax)

    # 若提供了手动范围，则覆盖自动范围（全局统一或按 field）
    plot_ylim: Dict[str, tuple[float, float]] = {}
    for field in field_names:
        if MANUAL_YLIM is None:
            plot_ylim[field] = field_ylim[field]
        elif isinstance(MANUAL_YLIM, tuple) and len(MANUAL_YLIM) == 2:
            plot_ylim[field] = (float(MANUAL_YLIM[0]), float(MANUAL_YLIM[1]))
        elif isinstance(MANUAL_YLIM, dict) and field in MANUAL_YLIM:
            ymin, ymax = MANUAL_YLIM[field]
            plot_ylim[field] = (float(ymin), float(ymax))
        else:
            plot_ylim[field] = field_ylim[field]

    # 为每个 (field, index) 创建独立 Buffer 和 DynamicMap
    window = min(200, total_steps)  # 可视窗口大小（点数）
    buffers: Dict[str, Dict[int, Buffer]] = {}
    dmaps: Dict[str, Dict[int, hv.DynamicMap]] = {}

    for field in field_names:
        buffers[field] = {}
        dmaps[field] = {}
        for idx in range(samples_per_field):
            buf = Buffer(np.empty((0, 2)), length=window)
            dmap = hv.DynamicMap(hv.Curve, streams=[buf]).opts(
                opts.Curve(
                    title=f"{field}[{idx}]",
                    width=350,
                    height=220,
                    line_width=2,
                    tools=["hover", "pan", "wheel_zoom", "reset"],
                    fontsize={"title": 11, "labels": 10, "xticks": 9, "yticks": 9},
                    framewise=False,  # 固定坐标范围，避免跳变
                    ylim=plot_ylim[field],
                )
            )
            buffers[field][idx] = buf
            dmaps[field][idx] = dmap

    # 组合为 NdLayout：两级 kdims -> (field, index) 网格布局
    layout_items = {}
    for field in field_names:
        for idx in range(samples_per_field):
            layout_items[(field, idx)] = dmaps[field][idx]

    ndlayout = hv.NdLayout(layout_items, kdims=["field", "index"]).cols(
        samples_per_field
    )

    # 在 Panel 中展示
    header = pn.pane.Markdown(
        "## Streaming by Field / Index\n同步滚动窗口，窗口大小：%d" % window
    )
    step_info = pn.pane.Markdown("Step: 0 / %d" % (total_steps - 1))
    dashboard = pn.Column(
        header, step_info, pn.panel(ndlayout, sizing_mode="stretch_both")
    )

    # 数据推送线程：将每个时间步的 (t, value) 发送到对应 Buffer
    def stream_data():
        print("开始推送数据流...")
        for step in range(total_steps):
            current_step[0] = step
            t = step  # 使用时间步作为时间轴；如需真实时间可换为 time.time()
            data_at_step = time_series_data[step]

            # 推入每个 (field, index)
            for field in field_names:
                values = data_at_step[field]
                for idx, v in enumerate(values):
                    buffers[field][idx].send(np.array([[t, v]]))

            step_info.object = f"Step: {step} / {total_steps - 1}"
            print(f"已推送时间步: {step}")
            time.sleep(0.8)
        print("数据流推送完毕。")

    print("正在启动流式可视化服务器...\n请稍候，图表将在浏览器中自动打开。")

    stream_thread = threading.Thread(target=stream_data, daemon=True)
    stream_thread.start()

    # 在独立线程中启动可视化服务，主线程等待数据流结束
    server = dashboard.show(title="Field/Index Streaming Visualization", threaded=True)
    try:
        # 保持主线程存活直到数据流结束
        stream_thread.join()
    finally:
        # 收尾：尝试关闭可视化服务（不同版本 API 可能略有差异，容错处理）
        try:
            server.stop()
        except Exception:
            pass

    print("程序结束。")


if __name__ == "__main__":
    main()
