//! Communication primitives for pi framework
//!
//! Provides channels, topics, and streams for inter-component communication.
//! Designed for both same-process (lock-free) and multi-process scenarios.

mod channel;
mod stream;
mod topic;

pub use channel::{bounded_channel, unbounded_channel, Channel, Receiver, Sender};
pub use stream::{Stream, StreamConfig, StreamHandle};
pub use topic::{Topic, TopicConfig};
