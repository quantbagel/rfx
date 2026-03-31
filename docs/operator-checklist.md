# Operator Checklist

Checklist before deploying a policy to real hardware.

## Preconditions

- Robot powered on and reachable (serial port or network).
- Robot config selected and validated (`rfx doctor`).
- Policy trained and saved as a self-describing directory.

## Pre-deploy checks

1. Policy loads without error:
   ```python
   loaded = rfx.load_policy("runs/my-policy")
   print(loaded.policy_type, loaded.robot_config)
   ```

2. Mock deploy succeeds:
   ```bash
   rfx deploy runs/my-policy --robot so101 --mock --duration 10
   ```

3. Robot config matches the policy's training config (state_dim, action_dim, joint ordering).

4. Control frequency is appropriate for the robot (`rate_hz` in config or `--rate-hz` flag).

## Deploy

```bash
rfx deploy runs/my-policy --robot so101 --duration 30
```

Monitor jitter stats in the output. High p95/p99 jitter indicates the control loop can't keep up at the requested rate.

## Post-deploy

- Review `SessionStats` output (iterations, overruns, jitter percentiles).
- If overruns > 0, consider lowering `rate_hz` or profiling the policy.
- Keep the policy directory for reproducibility.
