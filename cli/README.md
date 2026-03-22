# CLI

Home for user-facing command-line tools.

Planned commands:

- `cli/rfx.sh doctor`: environment and dependency checks
- `cli/rfx.sh doctor-so101`: SO101-specific environment/port checks
- `cli/rfx.sh so101-demo`: run a minimal SO101 hardware quickstart
- `cli/rfx.sh so101-bimanual`: run bimanual SO101 teleop using `rfx/configs/so101_bimanual.yaml`
- `cli/rfx.sh bootstrap`: local project setup
- `cli/rfx.sh check`: run monorepo quality checks
- `cli/rfx.sh register --url <dashboard> --api-key <key>`: create or refresh the robot record in the platform
- `cli/rfx.sh probe`: measure candidate AWS region latency directly from the current machine
- `cli/rfx.sh probe --regions us-east-1,us-west-2`: probe a custom AWS region set locally
- `cli/rfx.sh probe --url <dashboard> --api-key <key> --robot-id <id> --submit`: upload probe results back to the platform
- `cli/rfx.sh connect --url <dashboard> --api-key <key>`: keep a registered robot online with periodic heartbeats
- `cli/rfx.sh pkg-create <name>`: create a runtime package scaffold
- `cli/rfx.sh pkg-list`: list runtime packages
- `cli/rfx.sh run <pkg> <node>`: run a node
- `cli/rfx.sh launch <file>`: run a launch graph
- `cli/rfx.sh graph`: inspect active launch graph
- `cli/rfx.sh topic-list`: inspect declared runtime topics
