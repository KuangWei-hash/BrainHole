"""v10 候選架構的參數量與 MACs 估算（每個 640x16 區塊）。

用於設計階段比較不同寬度／深度的取捨。所有數字都是解析式估算，不含
pixel_shuffle、殘差加法等無參數操作，也沒有計入 bias、LayerNorm 與
注意力分支的逐元素運算。

幾何（由 training-data-preparation 的 Downsample/Up 推導）：
  低解析度 RGGB 打包平面   4 x 2 x 80      (= 160x4 低解析度馬賽克)
  高解析度 RGGB 打包平面   4 x 8 x 320     (= 640x16 高解析度馬賽克)
  每平面縮放比            4x（水平與垂直）
"""
import json

LR_H, LR_W = 2, 80
HR_H, HR_W = 8, 320


def macs(params_config):
    """回傳 (參數數, MACs/區塊) 。params_config 描述各段。"""
    channels = params_config["channels"]
    blocks = params_config["blocks"]
    phase_channels = params_config.get("phase_channels", 8)
    up_mode = params_config.get("up_mode", "shuffle4")
    stem_in = 4 + phase_channels
    macs_total = 0
    params_total = 0

    # stem: Conv3x3(stem_in -> C) @ LR
    macs_total += LR_H * LR_W * stem_in * channels * 9
    params_total += stem_in * channels * 9 + channels

    # body: N x NAFBlock(C) @ LR
    #   conv1 1x1 C->2C ; conv2 dw3x3 2C ; conv3 1x1 2C->C
    #   SCA: dw3x3 C ; 1x1 C->C
    #   gate: 1x1 C->C ; 1x1 C->C
    per_block = 2 * channels * channels + 18 * channels + 2 * channels * channels \
        + 9 * channels + channels * channels + channels * channels + channels * channels
    macs_total += blocks * LR_H * LR_W * per_block
    params_total += blocks * (2 * channels * channels + 18 * channels + 2 * channels * channels
                              + 9 * channels + 3 * channels * channels + 3 * channels)

    if up_mode == "shuffle4":
        # Conv3x3(C->C) @ LR ; 1x1(C->16C) @ LR ; PixelShuffle(4)
        macs_total += LR_H * LR_W * channels * channels * 9
        params_total += channels * channels * 9 + channels
        macs_total += LR_H * LR_W * channels * 16 * channels
        params_total += channels * 16 * channels + 16 * channels
    elif up_mode == "shuffle2x2":
        # Conv3x3(C->4C) @ LR ; PS(2) ; Conv3x3(C->4C) @ 2x ; PS(2)
        macs_total += LR_H * LR_W * channels * 4 * channels * 9
        params_total += channels * 4 * channels * 9 + 4 * channels
        macs_total += 4 * LR_H * LR_W * channels * 4 * channels * 9
        params_total += channels * 4 * channels * 9 + 4 * channels

    # tail: Conv3x3(C->4) @ HR
    macs_total += HR_H * HR_W * channels * 4 * 9
    params_total += channels * 4 * 9 + 4

    return params_total, macs_total


def v9_cost():
    """v9 plain16：兩層 Conv3x3（4->16->4）在高解析度平面網格 8x320。"""
    macs = 2 * HR_H * HR_W * 4 * 16 * 9
    params = 4 * 16 * 9 + 16 + 16 * 4 * 9 + 4
    return params, macs


def v7_cost():
    """v7：stem Conv3x3(21->24) + 5 個殘差區塊(24->48->24) + 輸出 conv，全部在高解析度網格。"""
    pos = HR_H * HR_W
    macs = pos * 21 * 24 * 9
    macs += 5 * 2 * pos * 24 * 48 * 9
    macs += pos * 24 * 4 * 9
    params = 21 * 24 * 9 + 24
    params += 5 * (24 * 48 * 9 + 48 + 48 * 24 * 9 + 24)
    params += 24 * 4 * 9 + 4
    return params, macs


def main():
    rows = []
    p, m = v9_cost()
    rows.append({"model": "v9 plain16 (現況基準)", "params": p, "macs_per_block": m,
                 "body_resolution": "HR 8x320", "receptive_field_mosaic_px": 2})
    p, m = v7_cost()
    rows.append({"model": "v7 (5 殘差區塊, 雙幀)", "params": p, "macs_per_block": m,
                 "body_resolution": "HR 8x320", "receptive_field_mosaic_px": 11})
    for channels in (16, 24, 32, 48):
        for blocks in (4, 8, 12):
            config = {"channels": channels, "blocks": blocks, "up_mode": "shuffle4"}
            p, m = macs(config)
            rows.append({"model": f"v10 C={channels} N={blocks}",
                         "params": p, "macs_per_block": m,
                         "body_resolution": "LR 2x80",
                         "receptive_field_mosaic_px": (2 * blocks + 1) * 4 // 2})
    print(json.dumps(rows, indent=2))
    print()
    print(f"{'model':<26}{'params':>10}{'MACs/block':>14}{'GMAC/s @900':>14}{'RF(+/- mosaic px)':>20}")
    for row in rows:
        print(f"{row['model']:<26}{row['params']:>10,}{row['macs_per_block']:>14,}"
              f"{row['macs_per_block']*900/1e9:>14.2f}{row['receptive_field_mosaic_px']:>20}")


if __name__ == "__main__":
    main()
