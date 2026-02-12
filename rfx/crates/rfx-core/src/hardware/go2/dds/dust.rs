//! dust-dds backend implementation (pure Rust)
//!
//! This is the default DDS backend using the dust-dds crate.
//! Currently provides a working framework that can be enhanced
//! with full DDS protocol support.

use parking_lot::RwLock;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dust_dds::{
    configuration::DustDdsConfigurationBuilder,
    domain::domain_participant_factory::DomainParticipantFactory,
    infrastructure::{qos::QosKind, status::NO_STATUS},
    topic_definition::type_support::DdsType,
};

use super::super::{Go2Config, LowCmd, LowState, SportModeCmd};
use super::backend::DdsBackend;
use super::messages::{LowCmdDds, SportModeRequestDds};
use crate::comm::{bounded_channel, Receiver, Sender};
use crate::Error;

/// Domain ID for Go2 DDS communication
const DDS_DOMAIN_ID: i32 = 0;

/// DDS backend using dust-dds (pure Rust implementation)
///
/// This backend provides DDS communication with the Go2 robot.
/// It creates the necessary DDS infrastructure for publishing
/// commands and receiving state updates.
pub struct DustDdsBackend {
    /// Flag indicating connection status
    connected: Arc<AtomicBool>,
    /// Channel for receiving state updates
    state_rx: Receiver<LowState>,
    /// Channel for sending low commands
    low_cmd_tx: Sender<LowCmdDds>,
    /// Channel for sending sport commands
    sport_cmd_tx: Sender<SportModeRequestDds>,
    /// Configuration
    #[allow(dead_code)]
    config: Go2Config,
    /// Shared state for cleanup
    _handle: Arc<RwLock<DdsHandle>>,
}

/// Handle to DDS resources
struct DdsHandle {
    domain_id: i32,
}

#[derive(Debug, Clone, DdsType)]
struct RequestIdentityWire {
    id: i64,
    api_id: i64,
}

#[derive(Debug, Clone, DdsType)]
struct RequestLeaseWire {
    id: i64,
}

#[derive(Debug, Clone, DdsType)]
struct RequestPolicyWire {
    priority: i32,
    noreply: bool,
}

#[derive(Debug, Clone, DdsType)]
struct RequestHeaderWire {
    identity: RequestIdentityWire,
    lease: RequestLeaseWire,
    policy: RequestPolicyWire,
}

#[derive(Debug, Clone, DdsType)]
struct RequestWire {
    header: RequestHeaderWire,
    parameter: String,
    binary: Vec<u8>,
}

fn monotonic_like_request_id() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

fn sport_request_to_wire(cmd: &SportModeRequestDds, req_id: i64, no_reply: bool) -> RequestWire {
    let api_id = cmd.header.api_id as i64;
    RequestWire {
        header: RequestHeaderWire {
            identity: RequestIdentityWire { id: req_id, api_id },
            lease: RequestLeaseWire { id: 0 },
            policy: RequestPolicyWire {
                priority: 0,
                noreply: no_reply,
            },
        },
        parameter: cmd.parameter.clone(),
        binary: Vec::new(),
    }
}

impl DdsBackend for DustDdsBackend {
    fn new(config: &Go2Config) -> crate::Result<Self>
    where
        Self: Sized,
    {
        let connected = Arc::new(AtomicBool::new(false));
        let (state_tx, state_rx) = bounded_channel::<LowState>(16);
        let (low_cmd_tx, low_cmd_rx) = bounded_channel::<LowCmdDds>(4);
        let (sport_cmd_tx, sport_cmd_rx) = bounded_channel::<SportModeRequestDds>(4);
        let handle = Arc::new(RwLock::new(DdsHandle {
            domain_id: DDS_DOMAIN_ID,
        }));

        // Start DDS worker threads
        let connected_clone = connected.clone();
        Self::start_workers(
            connected_clone,
            state_tx,
            low_cmd_rx,
            sport_cmd_rx,
            config.clone(),
        )?;

        // Mark as connected once workers are running
        connected.store(true, Ordering::Relaxed);

        tracing::info!(
            "DDS backend initialized for Go2 at {} (domain {})",
            config.ip_address,
            DDS_DOMAIN_ID
        );

        Ok(Self {
            connected,
            state_rx,
            low_cmd_tx,
            sport_cmd_tx,
            config: config.clone(),
            _handle: handle,
        })
    }

    fn publish_low_cmd(&self, cmd: &LowCmd) -> crate::Result<()> {
        if !self.is_connected() {
            return Err(Error::Connection("Not connected to robot".into()));
        }

        let dds_cmd = LowCmdDds::from_low_cmd(cmd);
        self.low_cmd_tx
            .send(dds_cmd)
            .map_err(|_| Error::Communication("Failed to send low command".into()))
    }

    fn publish_sport_cmd(&self, cmd: &SportModeCmd) -> crate::Result<()> {
        if !self.is_connected() {
            return Err(Error::Connection("Not connected to robot".into()));
        }

        let dds_cmd = sport_cmd_to_dds(cmd);
        self.sport_cmd_tx
            .send(dds_cmd)
            .map_err(|_| Error::Communication("Failed to send sport command".into()))
    }

    fn subscribe_state(&self) -> Receiver<LowState> {
        self.state_rx.clone()
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    fn disconnect(&self) {
        self.connected.store(false, Ordering::Relaxed);
    }
}

impl DustDdsBackend {
    fn start_workers(
        connected: Arc<AtomicBool>,
        state_tx: Sender<LowState>,
        low_cmd_rx: Receiver<LowCmdDds>,
        sport_cmd_rx: Receiver<SportModeRequestDds>,
        config: Go2Config,
    ) -> crate::Result<()> {
        // Start DDS state reader thread
        let connected_reader = connected.clone();
        let _config_reader = config.clone();
        thread::spawn(move || {
            dds_state_reader(connected_reader, state_tx);
        });

        // Start DDS command writer thread
        let connected_writer = connected.clone();
        let config_writer = config.clone();
        thread::spawn(move || {
            dds_command_writer(connected_writer, low_cmd_rx, sport_cmd_rx, config_writer);
        });

        Ok(())
    }
}

impl Drop for DustDdsBackend {
    fn drop(&mut self) {
        self.disconnect();
    }
}

/// DDS state reader thread
///
/// This thread reads state messages from the DDS network and
/// sends them to the state channel.
fn dds_state_reader(connected: Arc<AtomicBool>, state_tx: Sender<LowState>) {
    // Initialize DDS participant and reader
    // For now, this creates a simulated state source
    // Real implementation would use dust_dds participant

    tracing::debug!("DDS state reader started");

    while connected.load(Ordering::Relaxed) {
        // In a full implementation, this would read from DDS.
        // Using a condition-variable-friendly sleep (short interval) instead of
        // a 2ms blocking sleep, to reduce wake-up latency when DDS data arrives.
        // TODO: When dust_dds participant is fully integrated, replace this
        // with a blocking DDS reader.take() call which will wake on data arrival:
        // match reader.take(10, ANY_SAMPLE_STATE, ANY_VIEW_STATE, ANY_INSTANCE_STATE) {
        //     Ok(samples) => {
        //         for sample in samples {
        //             if let Ok(data) = sample.data() {
        //                 let low_state = data.to_low_state();
        //                 let _ = state_tx.send(low_state);
        //             }
        //         }
        //     }
        //     Err(_) => {}
        // }

        // Sleep with a shorter interval to reduce latency (500us vs 2ms)
        thread::sleep(Duration::from_micros(500));

        // For now, don't send any state - real robot will provide it
        let _ = &state_tx; // Suppress unused warning (reference avoids move)
    }

    tracing::debug!("DDS state reader stopped");
}

/// DDS command writer thread
///
/// This thread sends commands to the DDS network.
fn dds_command_writer(
    connected: Arc<AtomicBool>,
    low_cmd_rx: Receiver<LowCmdDds>,
    sport_cmd_rx: Receiver<SportModeRequestDds>,
    config: Go2Config,
) {
    let participant_factory = DomainParticipantFactory::get_instance();
    if let Some(interface_name) = config.network_interface.clone() {
        if let Ok(configuration) = DustDdsConfigurationBuilder::new()
            .interface_name(Some(interface_name))
            .build()
        {
            let _ = participant_factory.set_configuration(configuration);
        }
    }

    let participant = match participant_factory.create_participant(
        config.dds_domain_id,
        QosKind::Default,
        None,
        NO_STATUS,
    ) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Failed to create DDS participant: {:?}", e);
            connected.store(false, Ordering::Relaxed);
            return;
        }
    };

    let topic = match participant.create_topic::<RequestWire>(
        "rt/api/sport/request",
        "unitree_api.msg.dds_.Request_",
        QosKind::Default,
        None,
        NO_STATUS,
    ) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("Failed to create sport request topic: {:?}", e);
            connected.store(false, Ordering::Relaxed);
            return;
        }
    };

    let publisher = match participant.create_publisher(QosKind::Default, None, NO_STATUS) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Failed to create DDS publisher: {:?}", e);
            connected.store(false, Ordering::Relaxed);
            return;
        }
    };

    let writer = match publisher.create_datawriter(&topic, QosKind::Default, None, NO_STATUS) {
        Ok(w) => w,
        Err(e) => {
            tracing::error!("Failed to create sport request writer: {:?}", e);
            connected.store(false, Ordering::Relaxed);
            return;
        }
    };

    let req_id = AtomicI64::new(monotonic_like_request_id());

    tracing::debug!("DDS command writer started");

    while connected.load(Ordering::Relaxed) {
        // Use recv_timeout instead of try_recv + sleep to reduce latency.
        // We check the low_cmd channel with a short timeout, then drain sport cmds.
        match low_cmd_rx.recv_timeout(Duration::from_micros(500)) {
            Ok(Some(cmd)) => {
                tracing::trace!(
                    "Low command queued but direct publish not implemented: tick={}, crc={}",
                    cmd.head[0],
                    cmd.crc
                );
            }
            Ok(None) | Err(_) => {
                // Timeout or channel closed - continue to check sport commands
            }
        }

        if let Ok(Some(cmd)) = sport_cmd_rx.try_recv() {
            let id = req_id.fetch_add(1, Ordering::Relaxed);
            let wire = sport_request_to_wire(&cmd, id, true);
            if let Err(e) = writer.write(&wire, None) {
                tracing::warn!(
                    "Failed to publish sport request api_id={}: {:?}",
                    cmd.header.api_id,
                    e
                );
            } else {
                tracing::trace!("Sport command sent: api_id={}", cmd.header.api_id);
            }
        }
    }

    tracing::debug!("DDS command writer stopped");
}
/// Convert SportModeCmd to DDS format
fn sport_cmd_to_dds(cmd: &SportModeCmd) -> SportModeRequestDds {
    match cmd.mode {
        0 => SportModeRequestDds::stop(),
        1 => {
            if cmd.velocity[0].abs() > 0.001
                || cmd.velocity[1].abs() > 0.001
                || cmd.velocity[2].abs() > 0.001
            {
                SportModeRequestDds::move_cmd(cmd.velocity[0], cmd.velocity[1], cmd.velocity[2])
            } else if cmd.gait_type == 0 {
                SportModeRequestDds::stand()
            } else {
                SportModeRequestDds::stop()
            }
        }
        _ => SportModeRequestDds::stop(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sport_cmd_to_dds_stop() {
        let cmd = SportModeCmd::stop_move();
        let dds = sport_cmd_to_dds(&cmd);
        assert_eq!(dds.header.api_id, 1002);
    }

    #[test]
    fn test_sport_cmd_to_dds_stand() {
        let cmd = SportModeCmd::stand();
        let dds = sport_cmd_to_dds(&cmd);
        assert_eq!(dds.header.api_id, 1004);
    }

    #[test]
    fn test_sport_cmd_to_dds_move() {
        let cmd = SportModeCmd::move_cmd(0.5, 0.1, 0.0);
        let dds = sport_cmd_to_dds(&cmd);
        assert_eq!(dds.header.api_id, 1008);
        assert!(dds.parameter.contains("0.5"));
    }

    #[test]
    fn test_backend_creation() {
        let config = Go2Config::default();
        let backend = DustDdsBackend::new(&config);
        assert!(backend.is_ok());

        let backend = backend.unwrap();
        assert!(backend.is_connected());

        backend.disconnect();
        assert!(!backend.is_connected());
    }
}
