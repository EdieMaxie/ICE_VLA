"""Piper-X DM0.5 checkpoint-7000 adapter for the XPolicyLab WS server.

The training contract is three RGB cameras, 14 absolute joint/gripper values,
state conditioning, and a 50-step action chunk. Robot control remains client-side.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import PeftModel

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from opendm.constants.robot import RobotStateDesc
from opendm.exp.dm05_exp import DM05InferenceConfig, DM05ModelConfig
from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration
from .rtc_sampling import install as install_rtc
from .rtc_continuity import project as project_continuity

ARM_SPEED_SCALE = 0.5
ARM_INDEX = [0,1,2,3,4,5,7,8,9,10,11,12]
GRIP_INDEX = [6,13]


def retime_arm_and_gripper(chunk, anchor, preserve=0):
    """Half-speed arm forecast; delay gripper runs, retain their 25Hz samples.

    Operates once on a NEW forecast, never on the committed RTC prefix.
    A gripper run is consecutive nonzero increments with the same sign.
    This preserves commanded opening/closing slopes, not hardware feedback speed.
    """
    out=np.asarray(chunk,dtype=np.float32).copy()
    anchor=np.asarray(anchor,dtype=np.float32).reshape(14)
    if not 0<=preserve<len(out): raise ValueError('Invalid retiming prefix')
    nodes=np.vstack((out[preserve-1] if preserve else anchor,out[preserve:]))
    n=len(nodes)-1
    sample_times=np.arange(1,n+1)*ARM_SPEED_SCALE
    for j in ARM_INDEX:
        out[preserve:,j]=np.interp(sample_times,np.arange(n+1),nodes[:,j])
    for j in GRIP_INDEX:
        target=np.full(n,nodes[0,j],dtype=np.float32)
        increments=np.diff(nodes[:,j]);signs=np.sign(increments)
        i=0
        while i<n:
            if signs[i]==0: i+=1;continue
            end=i+1
            while end<n and signs[end]==signs[i]: end+=1
            begin=int(np.ceil((i+1)/ARM_SPEED_SCALE))-1
            if begin<n:
                count=min(end-i,n-begin)
                target[begin:begin+count]=nodes[i+1:i+1+count,j]
                target[begin+count:]=target[begin+count-1]
            i=end
        out[preserve:,j]=target
    return out


_HERE = Path(__file__).resolve().parent
_CAMERAS = {
    "images_1": ("cam_high", "cam_head", "head_camera", "top_camera"),
    "images_2": ("cam_left_wrist", "left_wrist", "left_camera", "wrist_left"),
    "images_3": ("cam_right_wrist", "right_wrist", "right_camera", "wrist_right"),
}
def _rgb_pil(observation: dict, image_key: str) -> Image.Image:
    vision = observation.get("vision") or {}
    for name in _CAMERAS[image_key]:
        if name not in vision:
            continue
        value = vision[name]
        if isinstance(value, dict):
            value = value.get("color", value.get("rgb"))
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim != 3:
            raise ValueError(f"{name}: expected RGB image, got {array.shape}")
        if array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] != 3:
            raise ValueError(f"{name}: expected three RGB channels, got {array.shape}")
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                raise ValueError(f"{name}: float RGB must be finite and in [0,1]")
            array = np.rint(array * 255).astype(np.uint8)
        elif np.issubdtype(array.dtype, np.integer):
            if array.min() < 0 or array.max() > 255:
                raise ValueError(f"{name}: integer RGB must be in [0,255]")
            array = array.astype(np.uint8, copy=False)
        else:
            raise TypeError(f"{name}: unsupported image dtype {array.dtype}")
        return Image.fromarray(np.ascontiguousarray(array), mode="RGB")
    raise KeyError(f"{image_key}: missing camera; expected one of {_CAMERAS[image_key]}")


def _instruction(obs: dict) -> str:
    prompt = obs.get("instruction") or obs.get("instructions") or obs.get("task")
    if isinstance(prompt, (tuple, list)):
        prompt = prompt[0] if prompt else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("DM05 requires observation.instruction; no Pi_05 bottle prompt fallback")
    return prompt.strip()


# RECAP Round-1 CFG training prefixed every conditional sample with
# "<task>\nAdvantage: positive" (RLinf dm05_cfg/data.py).  The launched config
# uses cfgrl_guidance_scale=1.0, which evaluates the positive branch only, so the
# deployed prompt must carry the identical suffix.  Without it the checkpoint is
# fed an out-of-distribution prefix and the guidance signal is silently unused.
_ADVANTAGE_CONDITIONS = frozenset({"positive", "negative"})


def _conditioned_prompt(prompt: str, condition: str | None) -> str:
    """Append the RECAP advantage condition exactly as the trainer wrote it."""
    if not condition:
        return prompt
    if condition not in _ADVANTAGE_CONDITIONS:
        raise ValueError(
            f"advantage_condition must be one of {sorted(_ADVANTAGE_CONDITIONS)}, "
            f"got {condition!r}"
        )
    return f"{prompt}\nAdvantage: {condition}"


class _PiperXModelConfig(DM05ModelConfig):
    """Load the LoRA base from the deployed directory, not its training path."""

    def __init__(self, base_path: str, adapter_path: str):
        super().__init__(model_name_or_path=adapter_path)
        self._deployment_base_path = base_path
        self.chunk_size = 50
        self.bf16 = True
        self.liger_kernel = False
        self.llm_attn_implementation = "sdpa"
        self.vision_attn_implementation = "sdpa"
        self.action_attn_implementation = "sdpa"
        self.vlm_gradient_checkpointing = False
        self.ae_gradient_checkpointing = False

    def _load_adapter_checkpoint_model(self):
        with open(Path(self.model_name_or_path) / "adapter_config.json", encoding="utf-8") as file:
            adapter_cfg = json.load(file)
        if adapter_cfg.get("peft_type") != "LORA" or adapter_cfg.get("r") != 32:
            raise ValueError("Expected official-structure rank-32 LoRA checkpoint")
        config = DM05Config.from_pretrained(self._deployment_base_path)
        for name, value in self._config_overrides().items():
            setattr(config, name, value)
        # Exact pre-construction SDPA override from dm05_piperx_lora_official.py.
        # The base config persists flash_attention_2, unavailable on this stack.
        config.vlm_config._attn_implementation = {
            "": self.llm_attn_implementation,
            "text_config": self.llm_attn_implementation,
            "vision_config": self.vision_attn_implementation,
        }
        config.vlm_config.vision_config._attn_implementation = self.vision_attn_implementation
        base = DM05ForConditionalGeneration.from_pretrained(
            self._deployment_base_path, config=config, dtype=self._torch_dtype()
        )
        return PeftModel.from_pretrained(base, self.model_name_or_path).merge_and_unload()


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict):
        if model_cfg.get("env_cfg_type") != "piper" or model_cfg.get("action_type") != "joint":
            raise ValueError("DM05 checkpoint-7000 requires env_cfg_type=piper, action_type=joint")
        self.dim_info = get_robot_action_dim_info("piper")
        if self.dim_info["arm_dim"] != [6, 6] or self.dim_info["ee_dim"] != [1, 1]:
            raise ValueError(f"Piper-X 6+1+6+1 contract mismatch: {self.dim_info}")
        if int(model_cfg.get("action_dim", 14)) != 14:
            raise ValueError("DM05 requires action_dim=14")
        self.action_type = "joint"
        self.action_steps = int(model_cfg.get("n_action_steps", 50))
        if self.action_steps != 50:
            raise ValueError("XPolicyLab DM05 must return the full trained 50-step chunk")

        # RECAP CFG adapters are trained on an advantage-conditioned prefix.
        # Absent/None keeps the plain SFT prompt contract.
        self.advantage_condition = model_cfg.get("advantage_condition")
        if self.advantage_condition is not None and (
            self.advantage_condition not in _ADVANTAGE_CONDITIONS
        ):
            raise ValueError(
                "advantage_condition must be one of "
                f"{sorted(_ADVANTAGE_CONDITIONS)}, got {self.advantage_condition!r}"
            )

        adapter = Path(model_cfg.get("model_path", _HERE / "checkpoints/checkpoint-7000")).resolve()
        base = Path(model_cfg.get("base_model_path", _HERE / "checkpoints/base")).resolve()
        for required in (adapter / "adapter_model.safetensors", adapter / "adapter_config.json",
                         adapter / "norm_stats.json", base / "model.safetensors", base / "config.json"):
            if not required.is_file():
                raise FileNotFoundError(required)
        self._observations = {}
        self._latest = {None: [0]}
        model = _PiperXModelConfig(str(base), str(adapter)).build_model(use_lora=False)
        inference = DM05InferenceConfig()
        inference.image_keys = list(_CAMERAS)
        inference.output_action_dim = 14
        inference.diffusion_steps = int(model_cfg.get("diffusion_steps", 10))
        inference.enable_bf16_compute = True
        inference._initialize(
            model=model,
            model_name_or_path=str(adapter),
            norm_stats_path=str(adapter / "norm_stats.json"),
            n_bins=256,
            model_max_length=1024,  # DM05TrainerConfig.model_max_length during SFT.
            use_absolute_action=False,  # Absolute training target needs no delta-to-absolute conversion.
            add_state=True,
            is_history=False,
        )
        self.inference = inference
        install_rtc(inference.model)
        self._rtc_stats = inference.norm_stats_file.select('PiperX')['action']
        self.state_desc = ([RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER]) * 2
        print(f"[DM05] adapter={adapter} base={base} cameras={list(_CAMERAS)} state/action=14 chunk=50 "
              f"advantage_condition={self.advantage_condition!r}", flush=True)

    @staticmethod
    def _scope(value: dict | None):
        return value.get("evaluation_id") if isinstance(value, dict) else None

    def update_obs(self, obs: dict):
        scope = self._scope(obs)
        idx = obs.get("env_idx", 0)
        self._observations[scope, idx] = obs
        self._latest[scope] = [idx]

    def update_obs_batch(self, obs_list: list[dict]):
        if not obs_list:
            raise ValueError("Empty observation batch")
        scope = self._scope(obs_list[0])
        indices = []
        for index, obs in enumerate(obs_list):
            if self._scope(obs) != scope:
                raise ValueError("Mixed evaluation scopes")
            idx = obs.get("env_idx", index)
            self._observations[scope, idx] = obs
            indices.append(idx)
        self._latest[scope] = indices

    def _predict(self, obs: dict, raw=False):
        state = pack_robot_state(obs, self.action_type, self.dim_info, source_type="obs").astype(np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(f"Expected finite 14-D Piper-X state, got {state.shape}")
        # opendm 的 PixelTransform 要求顶层 images 是一个 list，而非 images_1/2/3 三个独立键。
        payload = {"images": [_rgb_pil(obs, key) for key in _CAMERAS]}
        payload.update({
            "prompt": _conditioned_prompt(
                _instruction(obs), self.advantage_condition
            ),
            "state": state,
            "meta_data": {
                "robot_type": "PiperX",
                "speed": "0.5",
                "control_mode": None,
                "state_desc": self.state_desc,
                "valid_dim_mask": np.ones(14, dtype=bool),
            },
        })
        chunk = np.asarray(self.inference._predict(payload), dtype=np.float32)
        if chunk.shape != (50, 14) or not np.isfinite(chunk).all():
            raise ValueError(f"DM05 action shape/finite mismatch: {chunk.shape}")
        return chunk if raw else unpack_robot_state(chunk, self.action_type, self.dim_info, source_type="obs")

    def get_action(self, scope: dict | None = None):
        key = self._scope(scope)
        return self._predict(self._observations[key, self._latest[key][0]])

    def rtc_capabilities(self):
        return {'version': 1, 'chunk_size': 50, 'action_dim': 14,
                'mode': 'vjp_guided_flow_inpainting', 'control_hz': 25,
                'overlap': 30, 'commit': 18,
                'arm_speed_scale': ARM_SPEED_SCALE,
                'gripper_retime': 'delay_runs_keep_rate'}

    def rtc_infer(self, obs):
        """Atomic observation + prefix RPC; context never persists between calls."""
        request = obs.get('rtc', {})
        if request.get('speed_contract') != 'pi05-half-speed-v1':
            raise RuntimeError('DM05 deploy is stale: reload updated deploy before evaluation; no actions generated')
        prefix = np.asarray(request.get('prefix', []), dtype=np.float32)
        commit = int(request.get('commit', 0))
        model = self.inference.model
        start = time.perf_counter()
        if prefix.size:
            if prefix.ndim != 2 or prefix.shape[1] != 14 or not np.isfinite(prefix).all():
                raise ValueError('RTC prefix must be finite N x 14')
            if not 2 <= commit < len(prefix) <= 49:
                raise ValueError('RTC commit/prefix range invalid')
            lo = np.asarray(self._rtc_stats.q01, dtype=np.float32)
            hi = np.asarray(self._rtc_stats.q99, dtype=np.float32)
            # Exact inverse of Denormalize; do NOT clip already planned actions.
            norm = 2 * (prefix-lo) / (hi-lo+1e-6) - 1
            norm = np.where((lo == 0) & (hi == 0), 0, norm)
            model._dm05_rtc_context = (norm, commit)
        else:
            model._dm05_rtc_context = None
        try:
            chunk = self._predict(obs, raw=True)
            anchor = pack_robot_state(obs, self.action_type, self.dim_info, source_type='obs').astype(np.float32)
            projection = {}
            if prefix.size:
                chunk[:commit] = prefix[:commit]
                chunk = retime_arm_and_gripper(chunk, anchor, preserve=commit)
                grip_commands = chunk[:, GRIP_INDEX].copy()
                chunk, projection = project_continuity(chunk, prefix, commit)
                chunk[:, GRIP_INDEX] = grip_commands
            else:
                # Capture/warm the VJP graph while the robot has not started.
                # Discard warmup actions; preserve the initial sampled chunk.
                lo = np.asarray(self._rtc_stats.q01, dtype=np.float32)
                hi = np.asarray(self._rtc_stats.q99, dtype=np.float32)
                warm = 2 * (chunk[-30:]-lo) / (hi-lo+1e-6) - 1
                warm = np.where((lo == 0) & (hi == 0), 0, warm)
                model._dm05_rtc_context = (warm, 18)
                self._predict(obs, raw=True)
                model._dm05_rtc_context = None
                chunk = retime_arm_and_gripper(chunk, anchor)
            actions = unpack_robot_state(chunk, self.action_type, self.dim_info, source_type='obs')
            print(f"[DM05 RTC server] request={request['request_id']} start={request['start_step']} "
                  f"latency_ms={(time.perf_counter()-start)*1000:.1f} projection={projection}",flush=True)
            return {'actions': actions, 'start_step': int(request['start_step']),
                    'request_id': request['request_id'], 'commit': commit,
                    'latency_sec': time.perf_counter()-start,
                    'arm_speed_scale': ARM_SPEED_SCALE, **projection}
        except Exception as exc:
            print(f"[DM05 RTC server] rejected request={request.get('request_id')} "
                  f"elapsed_ms={(time.perf_counter()-start)*1000:.1f}: {exc}",flush=True)
            raise
        finally:
            model._dm05_rtc_context = None

    def get_action_batch(self, env_idx_list=None):
        if isinstance(env_idx_list, dict):
            scope = self._scope(env_idx_list)
            indices = env_idx_list.get("env_idx_list") or self._latest[scope]
        else:
            scope = None
            indices = env_idx_list or self._latest[scope]
        return [self._predict(self._observations[scope, idx]) for idx in indices]

    def reset(self):
        self._observations.clear()
        self._latest = {None: [0]}

    def reset_evaluation(self, scope: dict):
        key = self._scope(scope)
        if not key:
            raise ValueError("evaluation_id required")
        for item in list(self._observations):
            if item[0] == key:
                del self._observations[item]
        self._latest.pop(key, None)
