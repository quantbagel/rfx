//! Publish-subscribe topics for message distribution
//!
//! Simplified ROS-like topics for pi framework.
//! Supports multiple publishers and subscribers.

use crossbeam_channel as cc;
use parking_lot::RwLock;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::Arc;

use super::Receiver;
use crate::Result;

/// Configuration for a topic
#[derive(Debug, Clone)]
pub struct TopicConfig {
    /// Maximum number of messages to buffer per subscriber
    pub buffer_size: usize,
    /// Whether to keep the latest message for new subscribers
    pub latch: bool,
    /// Topic name for debugging/logging
    pub name: Arc<str>,
}

impl Default for TopicConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10,
            latch: false,
            name: Arc::from(""),
        }
    }
}

impl TopicConfig {
    /// Create a new topic config with the given name
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set the buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Enable latching (keep last message for new subscribers)
    pub fn latch(mut self, enable: bool) -> Self {
        self.latch = enable;
        self
    }
}

/// Internal state for a topic
struct TopicInner<T> {
    config: TopicConfig,
    subscribers: Vec<cc::Sender<T>>,
    latched_value: Option<T>,
}

impl<T: Clone + Send + 'static> TopicInner<T> {
    #[inline]
    fn do_publish(&mut self, message: T) {
        if self.config.latch {
            self.latched_value = Some(message.clone());
        }
        self.subscribers
            .retain(|tx| tx.try_send(message.clone()).is_ok());
    }
}

/// A publish-subscribe topic
///
/// Multiple publishers can send messages, and multiple subscribers
/// receive copies of all messages.
///
/// # Example
/// ```ignore
/// let topic = Topic::<String>::new("greetings");
///
/// // Subscribe
/// let sub = topic.subscribe();
///
/// // Publish
/// topic.publish("Hello!".to_string());
///
/// // Receive
/// let msg = sub.recv().unwrap();
/// ```
pub struct Topic<T> {
    inner: Arc<RwLock<TopicInner<T>>>,
}

impl<T: Clone + Send + 'static> Topic<T> {
    /// Create a new topic with default configuration
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self::with_config(TopicConfig::new(name))
    }

    /// Create a new topic with custom configuration
    pub fn with_config(config: TopicConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(TopicInner {
                config,
                subscribers: Vec::new(),
                latched_value: None,
            })),
        }
    }

    /// Get the topic name
    #[inline]
    pub fn name(&self) -> Arc<str> {
        self.inner.read().config.name.clone()
    }

    /// Publish a message to all subscribers
    #[inline]
    pub fn publish(&self, message: T) {
        // Early exit: no subscribers and no latching
        {
            let inner = self.inner.read();
            if inner.subscribers.is_empty() && !inner.config.latch {
                return;
            }
        }

        // Need write lock for retain + latch
        self.inner.write().do_publish(message);
    }

    /// Subscribe to the topic
    pub fn subscribe(&self) -> Receiver<T> {
        let mut inner = self.inner.write();
        let (tx, rx) = cc::bounded(inner.config.buffer_size);

        // Send latched value if available
        if let Some(ref latched) = inner.latched_value {
            let _ = tx.try_send(latched.clone());
        }

        inner.subscribers.push(tx);
        Receiver { inner: rx }
    }

    /// Get the number of active subscribers
    pub fn subscriber_count(&self) -> usize {
        self.inner.read().subscribers.len()
    }

    /// Get the latched value if available
    pub fn latched(&self) -> Option<T> {
        self.inner.read().latched_value.clone()
    }

    /// Create a publisher handle for this topic
    pub fn publisher(&self) -> Publisher<T> {
        Publisher {
            topic: self.inner.clone(),
        }
    }
}

impl<T: Clone + Send + 'static> Clone for Topic<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// A publisher handle for a topic
pub struct Publisher<T> {
    topic: Arc<RwLock<TopicInner<T>>>,
}

impl<T: Clone + Send + 'static> Publisher<T> {
    /// Publish a message
    #[inline]
    pub fn publish(&self, message: T) {
        self.topic.write().do_publish(message);
    }
}

impl<T: Clone + Send + 'static> Clone for Publisher<T> {
    fn clone(&self) -> Self {
        Self {
            topic: self.topic.clone(),
        }
    }
}

/// A typed topic that enforces serialization
pub struct TypedTopic<T: Serialize + DeserializeOwned + Clone + Send + 'static> {
    topic: Topic<T>,
}

impl<T: Serialize + DeserializeOwned + Clone + Send + 'static> TypedTopic<T> {
    /// Create a new typed topic
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self {
            topic: Topic::new(name),
        }
    }

    /// Publish a message
    pub fn publish(&self, message: T) {
        self.topic.publish(message);
    }

    /// Subscribe to the topic
    pub fn subscribe(&self) -> Receiver<T> {
        self.topic.subscribe()
    }

    /// Serialize and publish a message
    pub fn publish_json(&self, message: &T) -> Result<()> {
        // Just clone and publish, serialization can be done by transport layer
        self.topic.publish(message.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_topic_pubsub() {
        let topic = Topic::<i32>::new("test");
        let sub = topic.subscribe();

        topic.publish(42);

        let msg = sub.recv().unwrap();
        assert_eq!(msg, 42);
    }

    #[test]
    fn test_topic_multiple_subscribers() {
        let topic = Topic::<i32>::new("test");
        let sub1 = topic.subscribe();
        let sub2 = topic.subscribe();

        topic.publish(42);

        assert_eq!(sub1.recv().unwrap(), 42);
        assert_eq!(sub2.recv().unwrap(), 42);
    }

    #[test]
    fn test_topic_latching() {
        let topic = Topic::<i32>::with_config(TopicConfig::new("test").latch(true));

        topic.publish(42);

        // New subscriber should get latched value
        let sub = topic.subscribe();
        let msg = sub.recv_timeout(Duration::from_millis(100)).unwrap();
        assert_eq!(msg, Some(42));
    }

    #[test]
    fn test_topic_subscriber_count() {
        let topic = Topic::<i32>::new("test");
        assert_eq!(topic.subscriber_count(), 0);

        let _sub1 = topic.subscribe();
        assert_eq!(topic.subscriber_count(), 1);

        let _sub2 = topic.subscribe();
        assert_eq!(topic.subscriber_count(), 2);
    }

    #[test]
    fn test_publisher_handle() {
        let topic = Topic::<i32>::new("test");
        let pub1 = topic.publisher();
        let sub = topic.subscribe();

        pub1.publish(42);

        assert_eq!(sub.recv().unwrap(), 42);
    }

    #[test]
    fn test_topic_threaded() {
        let topic = Topic::<i32>::new("test");
        let sub = topic.subscribe();
        let pub_handle = topic.publisher();

        let handle = thread::spawn(move || {
            for i in 0..10 {
                pub_handle.publish(i);
                thread::sleep(Duration::from_millis(1));
            }
        });

        let mut received = Vec::new();
        for _ in 0..10 {
            if let Some(msg) = sub.recv_timeout(Duration::from_secs(1)).unwrap() {
                received.push(msg);
            }
        }

        handle.join().unwrap();
        assert_eq!(received.len(), 10);
    }
}
