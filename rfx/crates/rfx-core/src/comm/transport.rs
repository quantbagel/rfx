//! Keyed transport primitives with in-process backend.
//!
//! The transport API is Zenoh-like at the key-pattern level (`*`, `?`, `**`)
//! while keeping a lightweight in-process implementation for hot paths.

use crossbeam_channel as cc;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Transport message envelope.
#[derive(Debug, Clone)]
pub struct TransportEnvelope {
    /// Key expression route (e.g. `teleop/left/state`).
    pub key: Arc<str>,
    /// Monotonic sequence number assigned by the backend.
    pub sequence: u64,
    /// Wall-clock timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Payload bytes (Arc-backed for cheap clone fanout).
    pub payload: Arc<[u8]>,
    /// Optional metadata encoded as JSON text.
    pub metadata_json: Option<Arc<str>>,
}

impl TransportEnvelope {
    /// Construct a new transport envelope.
    pub fn new(
        key: impl Into<Arc<str>>,
        sequence: u64,
        payload: Arc<[u8]>,
        metadata_json: Option<Arc<str>>,
    ) -> Self {
        Self {
            key: key.into(),
            sequence,
            timestamp_ns: now_ns(),
            payload,
            metadata_json,
        }
    }
}

/// Subscription handle for receiving keyed envelopes.
#[derive(Debug)]
pub struct TransportSubscription {
    id: u64,
    pattern: Arc<str>,
    rx: cc::Receiver<TransportEnvelope>,
}

impl TransportSubscription {
    /// Unique subscription id.
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Key pattern used for this subscription.
    #[inline]
    pub fn pattern(&self) -> Arc<str> {
        self.pattern.clone()
    }

    /// Blocking receive.
    #[inline]
    pub fn recv(&self) -> Option<TransportEnvelope> {
        self.rx.recv().ok()
    }

    /// Receive with timeout.
    #[inline]
    pub fn recv_timeout(&self, timeout: Duration) -> Option<TransportEnvelope> {
        self.rx.recv_timeout(timeout).ok()
    }

    /// Non-blocking receive.
    #[inline]
    pub fn try_recv(&self) -> Option<TransportEnvelope> {
        self.rx.try_recv().ok()
    }

    /// Number of queued envelopes.
    #[inline]
    pub fn len(&self) -> usize {
        self.rx.len()
    }

    /// Whether queue is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rx.is_empty()
    }
}

struct SubscriptionEntry {
    id: u64,
    pattern: Arc<str>,
    tx: cc::Sender<TransportEnvelope>,
}

/// Transport backend trait for keyed pub/sub paths.
pub trait TransportBackend: Send + Sync {
    /// Publish a payload for a route key.
    fn publish(
        &self,
        key: &str,
        payload: Arc<[u8]>,
        metadata_json: Option<Arc<str>>,
    ) -> TransportEnvelope;

    /// Subscribe to a key pattern.
    fn subscribe(&self, pattern: &str, capacity: usize) -> TransportSubscription;

    /// Remove a subscription by id.
    fn unsubscribe(&self, subscription_id: u64) -> bool;

    /// Count active subscriptions.
    fn subscription_count(&self) -> usize;
}

/// In-process keyed transport backend.
pub struct InprocTransport {
    subscriptions: RwLock<Vec<SubscriptionEntry>>,
    next_seq: AtomicU64,
    next_sub_id: AtomicU64,
}

impl InprocTransport {
    /// Create a new in-process transport backend.
    pub fn new() -> Self {
        Self {
            subscriptions: RwLock::new(Vec::new()),
            next_seq: AtomicU64::new(0),
            next_sub_id: AtomicU64::new(1),
        }
    }

    /// Convenience publish helper for borrowed payload bytes.
    #[inline]
    pub fn publish_bytes(
        &self,
        key: &str,
        payload: &[u8],
        metadata_json: Option<&str>,
    ) -> TransportEnvelope {
        self.publish(
            key,
            Arc::<[u8]>::from(payload.to_vec().into_boxed_slice()),
            metadata_json.map(|s| Arc::<str>::from(s.to_owned())),
        )
    }
}

impl Default for InprocTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl TransportBackend for InprocTransport {
    fn publish(
        &self,
        key: &str,
        payload: Arc<[u8]>,
        metadata_json: Option<Arc<str>>,
    ) -> TransportEnvelope {
        let sequence = self.next_seq.fetch_add(1, Ordering::Relaxed) + 1;
        let envelope = TransportEnvelope::new(key.to_owned(), sequence, payload, metadata_json);

        let subscribers = self.subscriptions.read();
        let mut stale_sub_ids = Vec::new();
        for sub in &*subscribers {
            if !key_matches_pattern(key, &sub.pattern) {
                continue;
            }
            match sub.tx.try_send(envelope.clone()) {
                Ok(()) => {}
                Err(cc::TrySendError::Full(_)) => {
                    // Drop on backpressure; this transport prioritizes producer latency.
                }
                Err(cc::TrySendError::Disconnected(_)) => {
                    stale_sub_ids.push(sub.id);
                }
            }
        }
        drop(subscribers);

        if !stale_sub_ids.is_empty() {
            self.subscriptions
                .write()
                .retain(|s| !stale_sub_ids.contains(&s.id));
        }

        envelope
    }

    fn subscribe(&self, pattern: &str, capacity: usize) -> TransportSubscription {
        let id = self.next_sub_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = cc::bounded(capacity.max(1));
        let pattern_arc: Arc<str> = Arc::from(pattern.to_owned());
        self.subscriptions.write().push(SubscriptionEntry {
            id,
            pattern: pattern_arc.clone(),
            tx,
        });
        TransportSubscription {
            id,
            pattern: pattern_arc,
            rx,
        }
    }

    fn unsubscribe(&self, subscription_id: u64) -> bool {
        let mut subs = self.subscriptions.write();
        let before = subs.len();
        subs.retain(|s| s.id != subscription_id);
        before != subs.len()
    }

    fn subscription_count(&self) -> usize {
        self.subscriptions.read().len()
    }
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn key_matches_pattern(key: &str, pattern: &str) -> bool {
    let key_parts: Vec<&str> = key.split('/').filter(|s| !s.is_empty()).collect();
    let pat_parts: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
    match_path_parts(&key_parts, &pat_parts)
}

fn match_path_parts(key_parts: &[&str], pat_parts: &[&str]) -> bool {
    if pat_parts.is_empty() {
        return key_parts.is_empty();
    }
    if pat_parts[0] == "**" {
        if pat_parts.len() == 1 {
            return true;
        }
        for i in 0..=key_parts.len() {
            if match_path_parts(&key_parts[i..], &pat_parts[1..]) {
                return true;
            }
        }
        return false;
    }

    if key_parts.is_empty() {
        return false;
    }

    if !segment_matches(key_parts[0], pat_parts[0]) {
        return false;
    }
    match_path_parts(&key_parts[1..], &pat_parts[1..])
}

fn segment_matches(input: &str, pattern: &str) -> bool {
    wildcard_match(input.as_bytes(), pattern.as_bytes())
}

fn wildcard_match(input: &[u8], pattern: &[u8]) -> bool {
    let mut dp = vec![vec![false; input.len() + 1]; pattern.len() + 1];
    dp[0][0] = true;

    for i in 1..=pattern.len() {
        if pattern[i - 1] == b'*' {
            dp[i][0] = dp[i - 1][0];
        }
    }

    for i in 1..=pattern.len() {
        for j in 1..=input.len() {
            dp[i][j] = match pattern[i - 1] {
                b'?' => dp[i - 1][j - 1],
                b'*' => dp[i - 1][j] || dp[i][j - 1],
                c => c == input[j - 1] && dp[i - 1][j - 1],
            };
        }
    }

    dp[pattern.len()][input.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_wildcards() {
        assert!(segment_matches("left", "left"));
        assert!(segment_matches("left", "l*ft"));
        assert!(segment_matches("left", "l?ft"));
        assert!(!segment_matches("left", "r*"));
    }

    #[test]
    fn test_path_wildcards() {
        assert!(key_matches_pattern("teleop/left/state", "teleop/*/state"));
        assert!(key_matches_pattern("teleop/left/state/extra", "teleop/**"));
        assert!(key_matches_pattern(
            "teleop/left/state/extra",
            "teleop/**/extra"
        ));
        assert!(!key_matches_pattern("teleop/left/state", "teleop/right/*"));
    }

    #[test]
    fn test_publish_subscribe() {
        let transport = InprocTransport::new();
        let sub = transport.subscribe("teleop/**", 8);
        let env = transport.publish_bytes("teleop/left/state", b"{}", Some("{\"a\":1}"));
        let got = sub.recv_timeout(Duration::from_millis(100));
        assert!(got.is_some());
        let got = got.unwrap();
        assert_eq!(got.sequence, env.sequence);
        assert_eq!(got.key.as_ref(), "teleop/left/state");
        assert_eq!(got.payload.as_ref(), b"{}");
        assert_eq!(got.metadata_json.as_deref(), Some("{\"a\":1}"));
    }

    #[test]
    fn test_unsubscribe() {
        let transport = InprocTransport::new();
        let sub = transport.subscribe("teleop/**", 4);
        assert_eq!(transport.subscription_count(), 1);
        assert!(transport.unsubscribe(sub.id()));
        assert_eq!(transport.subscription_count(), 0);
    }
}
