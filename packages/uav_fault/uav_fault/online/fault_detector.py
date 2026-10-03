"""
在线故障检测推理引擎。

加载训练好的在线二分类模型（BiLSTM / 1D-CNN / Transformer），对滑动窗口数据执行实时推理。

注意:
- 模型输入通道顺序由模型目录中的 norm_stats.json 的 `columns` 决定。
- 一些通道在训练端为“派生列”（例如 rate_err_*、acc_norm 等）。推理端会在缺失时自动补齐，
  避免线上数据缺列导致输入分布漂移。

用法:
    from uav_fault.online.fault_detector import FaultDetector
    detector = FaultDetector(model_dir='assets/models/online_detector')
    result = detector.predict(window)  # window: list[dict], len >= 150
"""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from uav_fault.models.cnn.lstm import (
        OnlineBiLSTM,
        OnlineCNN1D,
        OnlineKGMSTCA,
        OnlineMSTCA,
        OnlineTransformer,
    )
except ModuleNotFoundError:
    from uav_fault.models.cnn.lstm import (
        OnlineBiLSTM,
        OnlineCNN1D,
        OnlineKGMSTCA,
        OnlineMSTCA,
        OnlineTransformer,
    )

LABEL_NAMES = ["normal", "fault"]


def _safe_float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return float(v)


_MODEL_REGISTRY = {
    "bilstm": OnlineBiLSTM,
    "cnn1d": OnlineCNN1D,
    "transformer": OnlineTransformer,
    "mstca": OnlineMSTCA,
    "kg_mstca": OnlineKGMSTCA,
}


class FaultDetector:
    """在线故障检测器，封装模型加载和推理逻辑。"""

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        window_size: int | None = 150,
        exclude_features: list[str] | None = None,
    ):
        model_dir = Path(model_dir)
        self.device = torch.device(device)

        # 加载配置
        with open(model_dir / "config.json") as f:
            self.config = json.load(f)

        # 加载归一化参数
        with open(model_dir / "norm_stats.json") as f:
            stats = json.load(f)
        self.columns = stats["columns"]
        self.mean = np.array(stats["mean"], dtype=np.float32)  # (C,)
        self.std = np.array(stats["std"], dtype=np.float32)  # (C,)
        self.exclude_features = set(exclude_features or [])
        self.exclude_indices = [
            idx
            for idx, name in enumerate(self.columns)
            if name in self.exclude_features
        ]
        # 若缓存里携带 window 配置，优先使用；否则使用构造参数。
        cfg_window = stats.get("window_size")
        if window_size is None or (isinstance(window_size, int) and window_size <= 0):
            self.window_size = (
                int(cfg_window) if isinstance(cfg_window, (int, float)) else 150
            )
        else:
            self.window_size = int(window_size)

        model_name = str(self.config.get("model", "bilstm"))
        model_cls = _MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            raise ValueError(
                f"不支持的在线模型类型: {model_name} (支持: {sorted(_MODEL_REGISTRY.keys())})"
            )

        # 构建并加载模型（结构需与训练端一致）
        model_kwargs = dict(
            in_channels=self.config["in_channels"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            num_classes=self.config["num_classes"],
            dropout=self.config.get("dropout", 0.2),
        )
        # 按通道名索引的架构（KG-MSTCA）需要知道当前 schema，才能同时支持
        # 31 通道的 rfly_seq_columns_v1 和 26 通道的 rfly_seq_columns_v2。
        if "columns" in inspect.signature(model_cls).parameters:
            model_kwargs["columns"] = list(self.columns)
        self.model = model_cls(**model_kwargs)
        try:
            state_dict = torch.load(
                model_dir / "best.pt", map_location=self.device, weights_only=True
            )
        except TypeError:
            # older torch without weights_only
            state_dict = torch.load(model_dir / "best.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"[FaultDetector] 模型加载完成: {model_name} | {n_params:,} 参数, device={self.device}",
            file=sys.stderr,
        )

        # 训练端使用 ffill 处理稀疏通道（VIBRATION）。推理端用“最后一次观测值”补齐缺失。
        self._sparse_cols = {"accel_vibration_metric", "gyro_vibration_metric"}
        self._last_sparse: dict[str, float] = {c: 0.0 for c in self._sparse_cols}

    def _get_value(
        self, row: dict, col_name: str, sparse_state: dict[str, float] | None = None
    ) -> float:
        """从单帧 row 取值；缺失时补齐派生列/稀疏列。"""
        sparse_state = self._last_sparse if sparse_state is None else sparse_state
        if col_name in row:
            v = _safe_float(row.get(col_name))
            if v is None:
                if col_name in self._sparse_cols:
                    return float(sparse_state.get(col_name, 0.0))
                return 0.0
            if col_name in self._sparse_cols:
                sparse_state[col_name] = float(v)
            return float(v)

        # ── 派生列：若线上未提供则补算 ──
        actuator_cols = [
            "actuator_ctrl_1",
            "actuator_ctrl_2",
            "actuator_ctrl_3",
            "actuator_ctrl_4",
            "actuator_out_1",
            "actuator_out_2",
            "actuator_out_3",
            "actuator_out_4",
        ]
        if col_name in {
            "actuator_mean",
            "actuator_std",
            "actuator_effort",
            "motor_cross_mean",
            "motor_cross_std",
            "motor_effort",
        }:
            values = [_safe_float(row.get(name)) for name in actuator_cols]
            if not any(v is not None for v in values):
                values = [_safe_float(row.get(f"motor{i}")) for i in range(1, 5)]
            values = [v for v in values if v is not None]
            if not values:
                return 0.0
            mean = float(sum(values) / len(values))
            if col_name in {"actuator_mean", "motor_cross_mean"}:
                return mean
            if col_name in {"actuator_std", "motor_cross_std"}:
                var = sum((x - mean) ** 2 for x in values) / float(len(values))
                return float(np.sqrt(var))
            az = _safe_float(row.get("accel_z"))
            acc_norm = None
            if az is not None:
                ax = _safe_float(row.get("accel_x"))
                ay = _safe_float(row.get("accel_y"))
                if ax is not None and ay is not None:
                    acc_norm = float(np.sqrt(ax * ax + ay * ay + az * az))
            denom = max(
                acc_norm
                if acc_norm is not None
                else abs(float(az))
                if az is not None
                else 0.0,
                0.1,
            )
            return float(mean / denom)

        if col_name == "acc_norm":
            ax = _safe_float(row.get("accel_x"))
            ay = _safe_float(row.get("accel_y"))
            az = _safe_float(row.get("accel_z"))
            if ax is None or ay is None or az is None:
                return 0.0
            return float(np.sqrt(ax * ax + ay * ay + az * az))

        if col_name.startswith("rate_err_"):
            if col_name == "rate_err_roll":
                sp = _safe_float(row.get("rates_sp_roll"))
                actual = _safe_float(row.get("rollspeed"))
            elif col_name == "rate_err_pitch":
                sp = _safe_float(row.get("rates_sp_pitch"))
                actual = _safe_float(row.get("pitchspeed"))
            else:  # rate_err_yaw
                sp = _safe_float(row.get("rates_sp_yaw"))
                actual = _safe_float(row.get("yawspeed"))
            if sp is None or actual is None:
                return 0.0
            return float(sp - actual)

        return 0.0

    def _build_feature_matrix(
        self, records, sparse_state: dict[str, float] | None = None
    ) -> np.ndarray:
        """将整段 records 转成未归一化特征矩阵 (T, C)。"""
        n_rows = len(records)
        data = np.zeros((n_rows, len(self.columns)), dtype=np.float32)
        sparse_state = self._last_sparse if sparse_state is None else sparse_state
        for t, row in enumerate(records):
            for c, col_name in enumerate(self.columns):
                data[t, c] = self._get_value(row, col_name, sparse_state=sparse_state)
        if self.exclude_indices:
            data[:, self.exclude_indices] = 0.0
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_feature_matrix(self, data: np.ndarray) -> np.ndarray:
        """按训练统计做 z-score 归一化。"""
        return (data - self.mean[np.newaxis, :]) / self.std[np.newaxis, :]

    def _window_end_indices(self, n_rows: int, every_n: int) -> np.ndarray:
        """返回与实时链路一致的窗口结束位置（1-based exclusive end）。"""
        if every_n <= 0:
            raise ValueError("every_n must be positive")
        if n_rows < self.window_size:
            return np.empty(0, dtype=np.int64)
        first_end = ((self.window_size + every_n - 1) // every_n) * every_n
        return np.arange(first_end, n_rows + 1, every_n, dtype=np.int64)

    def _predict_batch_tensor(self, batch_x: np.ndarray) -> list[dict]:
        """对 (B, C, T) batch 执行推理并返回与 predict 一致的结果。"""
        if batch_x.size == 0:
            return []
        x = torch.from_numpy(batch_x).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        logits_np = logits.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        outputs = []
        for logit_row, prob_row in zip(logits_np, probs_np):
            pred_idx = int(prob_row.argmax())
            outputs.append(
                {
                    "label": LABEL_NAMES[pred_idx],
                    "confidence": float(prob_row[pred_idx]),
                    "fault_prob": float(prob_row[1]),
                    "raw_logits": logit_row.tolist(),
                }
            )
        return outputs

    def predict_records_batch(
        self, records, every_n: int, batch_size: int = 256
    ) -> list[tuple[int, dict]]:
        """离线 replay helper：整文件提特征一次，按 batch 切窗推理。"""
        end_indices = self._window_end_indices(len(records), every_n)
        if end_indices.size == 0:
            return []
        features = self._normalize_feature_matrix(self._build_feature_matrix(records))
        outputs: list[tuple[int, dict]] = []
        for start in range(0, int(end_indices.size), int(batch_size)):
            batch_ends = end_indices[start : start + int(batch_size)]
            batch_x = np.stack(
                [features[end - self.window_size : end].T for end in batch_ends],
                axis=0,
            ).astype(np.float32, copy=False)
            preds = self._predict_batch_tensor(batch_x)
            outputs.extend(
                (int(end), pred) for end, pred in zip(batch_ends.tolist(), preds)
            )
        return outputs

    def predict(self, window) -> dict:
        """对滑动窗口执行故障检测推理。

        Args:
            window: list[dict] 或 deque[dict]，每个 dict 是一帧遥测数据。
                    长度需 >= self.window_size。

        Returns:
            dict: {
                'label': 'normal' 或 'fault',
                'confidence': float (预测类别的概率),
                'fault_prob': float (故障概率),
                'raw_logits': [float, float]
            }
            若窗口不满返回 None。
        """
        if len(window) < self.window_size:
            return None

        recent = list(window)[-self.window_size :]
        data = self._build_feature_matrix(recent, sparse_state=self._last_sparse)
        data = self._normalize_feature_matrix(data)
        preds = self._predict_batch_tensor(
            data.T[np.newaxis, :, :].astype(np.float32, copy=False)
        )
        return preds[0]
