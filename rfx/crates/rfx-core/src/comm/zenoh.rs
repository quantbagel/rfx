//! Zenoh-backed transport backend for cross-process pub/sub.
//!
//! Uses a bridge pattern: `ZenohTransport` keeps the `TransportBackend` trait
//! fully synchronous by calling Zenoh's sync `.wait()` methods under the hood.

use crossbeam_channel as cc;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use zenoh::bytes::ZBytes;
use zenoh::pubsub::Subscriber;
use zenoh::Wait;

use super::transport::{TransportBackend, TransportEnvelope, TransportSubscription};

/// Configuration for the Zenoh transport backend.
#[derive(Debug, Clone)]
pub struct ZenohTransportConfig {
    /// Zenoh endpoints to connect to (e.g. `["tcp/192.168.1.1:7447"]`).
    pub connect: Vec<String>,
    /// Zenoh endpoints to listen on (e.g. `["tcp/0.0.0.0:7447"]`).
    pub listen: Vec<String>,
    /// Enable shared-memory transport for same-machine zero-copy.
    pub shared_memory: bool,
    /// Optional prefix prepended to all key expressions.
    pub key_prefix: String,
}

impl Default for ZenohTransportConfig {
    fn default() -> Self {
        Self {
            connect: Vec::new(),
            listen: Vec::new(),
            shared_memory: true,
            key_prefix: String::new(),
        }
    }
}

/// Metadata wire format encoded as Zenoh attachment bytes.
#[derive(serde::Serialize, serde::Deserialize)]
struct AttachmentMeta {
    seq: u64,
    ts: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    meta: Option<String>,
}

/// Active Zenoh subscription entry.
struct ZenohSubscriptionEntry {
    id: u64,
    _pattern: Arc<str>,
    _tx: cc::Sender<TransportEnvelope>,
    /// Dropping this undeclares the Zenoh subscriber.
    _subscriber: Subscriber<()>,
}

/// Zenoh-backed keyed transport backend.
///
/// All operations are synchronous — Zenoh 1.7's builder API exposes `.wait()`
/// methods that perform the work on the calling thread.
pub struct ZenohTransport {
    session: zenoh::Session,
    subscriptions: RwLock<Vec<ZenohSubscriptionEntry>>,
    next_seq: AtomicU64,
    next_sub_id: AtomicU64,
    key_prefix: String,
}

impl ZenohTransport {
    /// Open a new Zenoh session with the given configuration.
    ///
    /// Blocks the caller while the session is established.
    pub fn new(config: ZenohTransportConfig) -> crate::Result<Self> {
        let mut zenoh_config = zenoh::Config::default();

        if !config.connect.is_empty() {
            let json_endpoints: Vec<String> = config
                .connect
                .iter()
                .map(|s| format!("\"{}\"", s))
                .collect();
            let json_array = format!("[{}]", json_endpoints.join(","));
            zenoh_config
                .insert_json5("connect/endpoints", &json_array)
                .map_err(|e| crate::Error::Config(format!("invalid connect endpoints: {e}")))?;
        }

        if !config.listen.is_empty() {
            let json_endpoints: Vec<String> =
                config.listen.iter().map(|s| format!("\"{}\"", s)).collect();
            let json_array = format!("[{}]", json_endpoints.join(","));
            zenoh_config
                .insert_json5("listen/endpoints", &json_array)
                .map_err(|e| crate::Error::Config(format!("invalid listen endpoints: {e}")))?;
        }

        if config.shared_memory {
            zenoh_config
                .insert_json5("transport/shared_memory/enabled", "true")
                .map_err(|e| {
                    crate::Error::Config(format!("failed to enable shared memory: {e}"))
                })?;
        }

        let session = zenoh::open(zenoh_config)
            .wait()
            .map_err(|e| crate::Error::Connection(format!("failed to open zenoh session: {e}")))?;

        Ok(Self {
            session,
            subscriptions: RwLock::new(Vec::new()),
            next_seq: AtomicU64::new(0),
            next_sub_id: AtomicU64::new(1),
            key_prefix: config.key_prefix,
        })
    }

    /// Build a full key expression with the configured prefix.
    fn full_key(&self, key: &str) -> String {
        if self.key_prefix.is_empty() {
            key.to_owned()
        } else {
            format!("{}/{}", self.key_prefix, key)
        }
    }
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

impl TransportBackend for ZenohTransport {
    fn publish(
        &self,
        key: &str,
        payload: Arc<[u8]>,
        metadata_json: Option<Arc<str>>,
    ) -> TransportEnvelope {
        let sequence = self.next_seq.fetch_add(1, Ordering::Relaxed) + 1;
        let timestamp_ns = now_ns();
        let full_key = self.full_key(key);

        let attachment = AttachmentMeta {
            seq: sequence,
            ts: timestamp_ns,
            meta: metadata_json.as_ref().map(|s| s.to_string()),
        };
        let attachment_bytes = serde_json::to_vec(&attachment).unwrap_or_default();

        let payload_zbytes = ZBytes::from(payload.to_vec());
        let attachment_zbytes = ZBytes::from(attachment_bytes);

        // Synchronous fire-and-forget publish.
        let _ = self
            .session
            .put(&full_key, payload_zbytes)
            .attachment(attachment_zbytes)
            .wait();

        TransportEnvelope {
            key: Arc::from(key.to_owned()),
            sequence,
            timestamp_ns,
            payload,
            metadata_json,
        }
    }

    fn subscribe(&self, pattern: &str, capacity: usize) -> TransportSubscription {
        let id = self.next_sub_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = cc::bounded(capacity.max(1));
        let pattern_arc: Arc<str> = Arc::from(pattern.to_owned());
        let full_pattern = self.full_key(pattern);
        let prefix_len = if self.key_prefix.is_empty() {
            0
        } else {
            self.key_prefix.len() + 1 // +1 for the '/'
        };

        let tx_clone = tx.clone();
        let subscriber = self
            .session
            .declare_subscriber(&full_pattern)
            .callback(move |sample| {
                let raw_key = sample.key_expr().as_str();
                // Strip the key prefix to give callers the original key.
                let key: Arc<str> = if prefix_len > 0 && raw_key.len() > prefix_len {
                    Arc::from(&raw_key[prefix_len..])
                } else {
                    Arc::from(raw_key)
                };

                let payload_bytes: Vec<u8> = sample.payload().to_bytes().to_vec();

                let (sequence, timestamp_ns, metadata_json) =
                    if let Some(attachment) = sample.attachment() {
                        let att_bytes: Vec<u8> = attachment.to_bytes().to_vec();
                        match serde_json::from_slice::<AttachmentMeta>(&att_bytes) {
                            Ok(att) => (att.seq, att.ts, att.meta.map(Arc::<str>::from)),
                            Err(_) => (0, now_ns(), None),
                        }
                    } else {
                        (0, now_ns(), None)
                    };

                let envelope = TransportEnvelope {
                    key,
                    sequence,
                    timestamp_ns,
                    payload: Arc::from(payload_bytes.into_boxed_slice()),
                    metadata_json,
                };

                // Drop on backpressure — matches InprocTransport behavior.
                let _ = tx_clone.try_send(envelope);
            })
            .wait()
            .expect("failed to declare zenoh subscriber");

        self.subscriptions.write().push(ZenohSubscriptionEntry {
            id,
            _pattern: pattern_arc.clone(),
            _tx: tx,
            _subscriber: subscriber,
        });

        TransportSubscription::new(id, pattern_arc, rx)
    }

    fn unsubscribe(&self, subscription_id: u64) -> bool {
        let mut subs = self.subscriptions.write();
        let before = subs.len();
        // Dropping the entry also drops the `_subscriber` handle,
        // which automatically undeclares the Zenoh subscriber.
        subs.retain(|s| s.id != subscription_id);
        before != subs.len()
    }

    fn subscription_count(&self) -> usize {
        self.subscriptions.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zenoh_config_defaults() {
        let config = ZenohTransportConfig::default();
        assert!(config.connect.is_empty());
        assert!(config.listen.is_empty());
        assert!(config.shared_memory);
        assert!(config.key_prefix.is_empty());
    }

    #[test]
    fn test_zenoh_config_custom() {
        let config = ZenohTransportConfig {
            connect: vec!["tcp/192.168.1.1:7447".into()],
            listen: vec!["tcp/0.0.0.0:7447".into()],
            shared_memory: false,
            key_prefix: "rfx/robot1".into(),
        };
        assert_eq!(config.connect.len(), 1);
        assert!(!config.shared_memory);
        assert_eq!(config.key_prefix, "rfx/robot1");
    }

    #[test]
    fn test_attachment_meta_roundtrip() {
        let meta = AttachmentMeta {
            seq: 42,
            ts: 1234567890,
            meta: Some(r#"{"x":1}"#.into()),
        };
        let bytes = serde_json::to_vec(&meta).unwrap();
        let decoded: AttachmentMeta = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded.seq, 42);
        assert_eq!(decoded.ts, 1234567890);
        assert_eq!(decoded.meta.as_deref(), Some(r#"{"x":1}"#));
    }

    #[test]
    fn test_attachment_meta_without_metadata() {
        let meta = AttachmentMeta {
            seq: 1,
            ts: 0,
            meta: None,
        };
        let json = serde_json::to_string(&meta).unwrap();
        assert!(!json.contains("meta"));
        let decoded: AttachmentMeta = serde_json::from_str(&json).unwrap();
        assert!(decoded.meta.is_none());
    }

    // Integration tests that require Zenoh peer connectivity.
    #[test]
    #[ignore]
    fn test_zenoh_pub_sub_roundtrip() {
        let config = ZenohTransportConfig::default();
        let transport = ZenohTransport::new(config).expect("failed to open zenoh session");

        let sub = transport.subscribe("test/**", 16);
        let _env = transport.publish(
            "test/hello",
            Arc::from(b"world".to_vec().into_boxed_slice()),
            Some(Arc::from(r#"{"k":"v"}"#.to_owned())),
        );

        let got = sub.recv_timeout(std::time::Duration::from_secs(2));
        assert!(got.is_some(), "expected to receive a message");
        let got = got.unwrap();
        assert_eq!(got.key.as_ref(), "test/hello");
        assert_eq!(got.payload.as_ref(), b"world");
        assert_eq!(got.metadata_json.as_deref(), Some(r#"{"k":"v"}"#));
    }

    #[test]
    #[ignore]
    fn test_zenoh_unsubscribe() {
        let config = ZenohTransportConfig::default();
        let transport = ZenohTransport::new(config).expect("failed to open zenoh session");

        let sub = transport.subscribe("test/**", 4);
        assert_eq!(transport.subscription_count(), 1);
        let sub_id = sub.id();
        assert!(transport.unsubscribe(sub_id));
        assert_eq!(transport.subscription_count(), 0);
    }

    #[test]
    #[ignore]
    fn test_zenoh_metadata_roundtrip() {
        let config = ZenohTransportConfig::default();
        let transport = ZenohTransport::new(config).expect("failed to open zenoh session");

        let sub = transport.subscribe("meta/**", 16);
        let published = transport.publish(
            "meta/test",
            Arc::from(b"data".to_vec().into_boxed_slice()),
            Some(Arc::from(r#"{"robot":"so101","arm":"left"}"#.to_owned())),
        );

        let got = sub.recv_timeout(std::time::Duration::from_secs(2));
        assert!(got.is_some());
        let got = got.unwrap();
        assert_eq!(got.sequence, published.sequence);
        assert_eq!(
            got.metadata_json.as_deref(),
            Some(r#"{"robot":"so101","arm":"left"}"#)
        );
    }
}
