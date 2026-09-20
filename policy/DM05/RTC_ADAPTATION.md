# DM05 异步推理与连续性适配交接

日期：2026-09-19。只改 DM05 推理适配与 deploy；未运行真实机器人动作。

## 部署位置与不变项

- L20：`/root/zjh/XPolicyLab/policy/DM05/`，端口仍为 **6001**。
- 评测机：`/home/user/.xrobot/XPolicyLab/policy/DM05/deploy.py`。
- L20 改动：`model.py`、`deploy.py`；新增 `rtc_sampling.py`、`rtc_continuity.py`。
- 权重仍为 checkpoint-7000；三相机/RGB、state/action顺序、归一化、14D绝对动作、50步chunk、10步去噪不变。
- 没有修改 OpenDM 公共源码、训练代码、机器人驱动、TASK_ENV.take_action、相机/夹爪标定、协议公共实现或 deploy.yml。
- SciPy/threadpoolctl 均使用已有环境包，没有重新安装环境。

## 新执行方式

1. 客户端单后台线程独占 WebSocket RPC；主线程独占 TASK_ENV。每步同步上传不再占用动作执行路径。
2. 目标25Hz、每段50步，剩30步时提前请求下一段。保留18步提交前缀，剩余12步软约束。
3. 服务端 `rtc_infer` 原子接收新观测、旧动作前缀、request_id、绝对动作索引。收到结果后丢弃已经执行的前缀，不重放、不赶帧。
4. 采用推理期 RTC：对预测干净动作 `x-t*v` 计算完整VJP梯度引导；权重上限10，前缀为硬区+五次多项式软区。最后投影回前缀约束。
5. 梯度推理使用CUDA Graph，复用OpenDM的KV补零和mask逻辑，将推理KV缓存固定到已配置的1024-token上限，避免状态token跨桶时再次捕获；首次动作下发前预热。图执行/普通执行做bf16误差校验，并在原始动作单位重新检查连续性。
6. 对尚未执行的后缀解最小改动的凸二次优化问题，约束位置差分和二阶差分。已承诺前缀完全不改。

这是适配到DM05的RTC实现，**不是官方原封不动的开关**：五次软权重、末端投影、二次优化是本适配的额外连续性约束。旧 `get_action()` 和batch执行路径保留；单机器人新版deploy调用RTC方法。

## 连续性与异常规则

- 关节步长上限：0.12 rad/step；关节二阶差分上限：0.012 rad/step²。
- 夹爪读数步长上限0.15、二阶差分0.05，维持原来的夹爪数值语义，不做反转。
- 接缝检查的是**速度变化**：已有运动不应在接缝被强迫突然减速。不是要求所有运动时的相邻关节位置差都很小。
- 修正与原预测相差超过0.15 rad（夹爪0.3）时拒收，避免将不同策略硬拼成一条路径。
- 18步/25Hz提供720ms时限；迟到、无效数值、请求ID错位、队列耗尽或约束失败都会停止下发并报错，不偷偷重放过期动作。
- 这些是推理侧验收约束，**不是经过硬件认证的安全极限**。实际停止/保持由原TASK_ENV及控制器负责。本改动不替代现场急停和监督。

## 验收与证据边界

- 本地8项测试：RPC单线程、动作索引对齐、截止时间、跳变拒绝、任务结束、回复ID、约束投影、已有速度不在接缝被突然制动。
- 保存观测+真实checkpoint的前向测试已执行；固定前缀误差为0，热态前向约0.46秒。
- 最终版真实网络回放100步通过：动作间隔中位数40.00ms、P95 40.42ms、最大50.47ms；请求返回约560–600ms，服务端前向/处理约462–477ms。
- 最终版L20本机WebSocket/JPEG回放100步通过：中位数40.00ms、P95 40.47ms、最大49.27ms；服务端处理约461–462ms。该项验证编码/解码链路，跨机延迟以前一项为准。
- 真实评测机到L20的网络回放、JPEG编码输入测试使用 `e2e_saved_obs.py`：`take_action`仅记录数组，**不会调用机器人驱动**。最终结果存为 `rtc_e2e.json` / `rtc_e2e_encoded.json`。
- 中间版本曾触发600ms截止，最终提高到720ms；跨token长度新建图导致的尖峰改为固定1024缓存。扩展保存观测回放曾触发0.1819rad预测修正保护，该限制未放宽；最终两项100步测试不能排除其他场景仍会触发它。
- 保存观测不是根据本次模型动作产生的闭环观测，因此这些测试验证工程链路/连续性保护，不证明任务成功率或任意场景无中断。
- 先前发现的首次定位、夹爪读数和标定差异未在本次修复，不能用平滑掩盖这些问题。

## 如何使用和查看

服务仍使用 `8.136.122.30:6001`，模型目录名仍为 `DM05`。

请结束旧的评测会话，重新创建DM05评测任务以加载新版deploy。客户端应出现：

```text
[DM05 RTC] request=... age_ms=... dropped=... model_ms=...
```

`dropped` 是按时间/已执行动作索引跳过的旧前缀，不是丢失控制指令。服务器启动日志：

```bash
tail -f /root/zjh/XPolicyLab/policy/DM05/dm05_rtc_server_20260919.log
```

第一次运行先由现场人员监督空载/低风险场景，不在运动中热切换策略。

## 回滚

先结束评测任务，确认没有6001连接，再回滚两端。

L20备份：`/root/zjh/XPolicyLab/policy/DM05/rtc_backup_20260919T085503Z/`

```bash
cd /root/zjh/XPolicyLab/policy/DM05
cp rtc_backup_20260919T085503Z/model.py model.py
cp rtc_backup_20260919T085503Z/deploy.py deploy.py
```

随后核对DM05服务PID，只停止DM05进程并用原 `bash run_server.sh` 重启，不停止其他策略。

评测机备份：

```bash
cd /home/user/.xrobot/XPolicyLab/policy/DM05
cp rtc_backup_20260919T085507Z/deploy.py deploy.py
```

新增helper文件可保留，旧model不会导入它们；无需删除权重或数据。重新创建评测任务以加载旧deploy。

参考：[PI RTC](https://www.pi.website/research/real_time_chunking)、[LeRobot RTC说明](https://huggingface.co/docs/lerobot/main/rtc)。
