//! Zenoh-bridged DDS backend for Go2 communication
//!
//! Uses zenoh-bridge-dds to translate between Zenoh and DDS protocols.
//! The Go2 firmware speaks DDS natively; the bridge handles protocol translation.
//!
//! ```text
//! Go2 firmware ←DDS(CDR)→ zenoh-bridge-dds ←Zenoh→ ZenohDdsBackend (in rfx)
//! ```

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use zenoh::bytes::ZBytes;
use zenoh::Wait;

use super::backend::DdsBackend;
use super::dust::{monotonic_like_request_id, sport_cmd_to_dds, sport_request_to_cdr};
use super::messages::{LowCmdDds, LowStateDds};

use crate::comm::{bounded_channel, Receiver, Sender};
use crate::hardware::go2::{Go2Config, LowCmd, LowState, SportModeCmd};
use crate::Error;

/// DDS backend that communicates via Zenoh, relying on zenoh-bridge-dds
/// to translate between Zenoh and the Go2's native DDS protocol.
pub struct ZenohDdsBackend {
    connected: Arc<AtomicBool>,
    state_rx: Receiver<LowState>,
    low_cmd_tx: Sender<LowCmdDds>,
    sport_cmd_tx: Sender<super::messages::SportModeRequestDds>,
}

impl DdsBackend for ZenohDdsBackend {
    fn new(config: &Go2Config) -> crate::Result<Self>
    where
        Self: Sized,
    {
        let connected = Arc::new(AtomicBool::new(false));
        let (state_tx, state_rx) = bounded_channel::<LowState>(16);
        let (low_cmd_tx, low_cmd_rx) = bounded_channel::<LowCmdDds>(4);
        let (sport_cmd_tx, sport_cmd_rx) =
            bounded_channel::<super::messages::SportModeRequestDds>(4);

        // Open Zenoh session
        let mut zenoh_config = zenoh::Config::default();
        if let Some(ref endpoint) = config.zenoh_endpoint {
            let json = format!("[\"{endpoint}\"]");
            zenoh_config
                .insert_json5("connect/endpoints", &json)
                .map_err(|e| Error::Config(format!("invalid zenoh endpoint: {e}")))?;
        }

        let session = Arc::new(
            zenoh::open(zenoh_config)
                .wait()
                .map_err(|e| Error::Connection(format!("zenoh session failed: {e}")))?,
        );

        // Start state reader thread
        let connected_r = connected.clone();
        let session_r = session.clone();
        thread::spawn(move || {
            zenoh_state_reader(connected_r, state_tx, session_r);
        });

        // Start command writer thread
        let connected_w = connected.clone();
        let session_w = session.clone();
        thread::spawn(move || {
            zenoh_command_writer(connected_w, low_cmd_rx, sport_cmd_rx, session_w);
        });

        connected.store(true, Ordering::Relaxed);

        tracing::info!(
            "ZenohDdsBackend initialized for Go2 (endpoint: {:?})",
            config.zenoh_endpoint
        );

        Ok(Self {
            connected,
            state_rx,
            low_cmd_tx,
            sport_cmd_tx,
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

impl Drop for ZenohDdsBackend {
    fn drop(&mut self) {
        self.disconnect();
    }
}

/// State reader thread: subscribes to `rt/lowstate` via Zenoh,
/// deserializes CDR bytes into LowState, and sends to the state channel.
fn zenoh_state_reader(
    connected: Arc<AtomicBool>,
    state_tx: Sender<LowState>,
    session: Arc<zenoh::Session>,
) {
    use zenoh::handlers::FifoChannel;

    let subscriber = match session
        .declare_subscriber("rt/lowstate")
        .with(FifoChannel::new(4))
        .wait()
    {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("Failed to subscribe to rt/lowstate: {e}");
            connected.store(false, Ordering::Relaxed);
            return;
        }
    };

    tracing::debug!("Zenoh state reader started on rt/lowstate");

    while connected.load(Ordering::Relaxed) {
        let sample = match subscriber.recv_timeout(Duration::from_millis(100)) {
            Ok(Some(s)) => s,
            Ok(None) | Err(_) => continue,
        };

        let payload: Vec<u8> = sample.payload().to_bytes().to_vec();
        match LowStateDds::from_cdr_bytes(&payload) {
            Ok(dds_state) => {
                let low_state = dds_state.to_low_state();
                let _ = state_tx.send(low_state);
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to parse LowState: {e} (payload len={})",
                    payload.len()
                );
            }
        }
    }

    tracing::debug!("Zenoh state reader stopped");
}

/// Command writer thread: receives commands from channels, serializes to CDR,
/// and publishes via Zenoh to the bridge.
fn zenoh_command_writer(
    connected: Arc<AtomicBool>,
    low_cmd_rx: Receiver<LowCmdDds>,
    sport_cmd_rx: Receiver<super::messages::SportModeRequestDds>,
    session: Arc<zenoh::Session>,
) {
    let req_id = AtomicI64::new(monotonic_like_request_id());

    tracing::debug!("Zenoh command writer started");

    while connected.load(Ordering::Relaxed) {
        // Check for low commands
        if let Ok(Some(cmd)) = low_cmd_rx.recv_timeout(Duration::from_micros(500)) {
            let cdr_bytes = cmd.to_cdr_bytes();
            if let Err(e) = session
                .put("rt/lowcmd", ZBytes::from(cdr_bytes))
                .wait()
            {
                tracing::warn!("Failed to publish low cmd via Zenoh: {e}");
            }
        }

        // Check for sport commands
        if let Ok(Some(cmd)) = sport_cmd_rx.try_recv() {
            let id = req_id.fetch_add(1, Ordering::Relaxed);
            match sport_request_to_cdr(&cmd, id) {
                Ok(cdr_bytes) => {
                    if let Err(e) = session
                        .put("rt/api/sport/request", ZBytes::from(cdr_bytes))
                        .wait()
                    {
                        tracing::warn!("Failed to publish sport request via Zenoh: {e}");
                    } else {
                        tracing::trace!(
                            "Sport command sent via Zenoh: api_id={}",
                            cmd.header.api_id
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to serialize sport request: {e}");
                }
            }
        }
    }

    tracing::debug!("Zenoh command writer stopped");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::go2::Go2BackendHint;

    #[test]
    fn test_config_with_zenoh_endpoint() {
        let config = Go2Config::new("192.168.123.161")
            .with_zenoh_endpoint("tcp/192.168.123.161:7447");

        assert_eq!(
            config.zenoh_endpoint.as_deref(),
            Some("tcp/192.168.123.161:7447")
        );
    }

    #[test]
    fn test_backend_hint_values() {
        assert_eq!(Go2BackendHint::Zenoh, Go2BackendHint::Zenoh);
        assert_ne!(Go2BackendHint::Zenoh, Go2BackendHint::DustDds);
        assert_ne!(Go2BackendHint::CycloneDds, Go2BackendHint::DustDds);
    }

    #[test]
    fn test_config_with_backend() {
        let config = Go2Config::new("192.168.123.161")
            .with_backend(Go2BackendHint::Zenoh);

        assert_eq!(config.preferred_backend, Some(Go2BackendHint::Zenoh));
    }

    #[test]
    fn test_config_defaults_none() {
        let config = Go2Config::default();
        assert!(config.zenoh_endpoint.is_none());
        assert!(config.preferred_backend.is_none());
    }

    // Integration tests requiring zenoh-bridge-dds + robot
    #[test]
    #[ignore]
    fn test_zenoh_dds_state_subscription() {
        let config =
            Go2Config::new("192.168.123.161").with_backend(Go2BackendHint::Zenoh);
        let backend = ZenohDdsBackend::new(&config).expect("failed to create backend");
        let state_rx = backend.subscribe_state();

        // Wait for a state update (requires bridge + robot)
        match state_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Some(state)) => {
                println!("Got state: tick={}", state.tick);
                assert!(state.tick > 0);
            }
            _ => panic!("No state received within 5s"),
        }

        backend.disconnect();
    }

    #[test]
    #[ignore]
    fn test_zenoh_dds_sport_command() {
        let config =
            Go2Config::new("192.168.123.161").with_backend(Go2BackendHint::Zenoh);
        let backend = ZenohDdsBackend::new(&config).expect("failed to create backend");

        let cmd = SportModeCmd::stand();
        backend
            .publish_sport_cmd(&cmd)
            .expect("failed to publish sport cmd");

        backend.disconnect();
    }
}
