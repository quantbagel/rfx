//! Typed streams with backpressure support
//!
//! Streams provide a higher-level abstraction over channels with
//! features like transformation, filtering, and batching.

use crossbeam_channel as cc;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::Receiver;

/// Configuration for a stream
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Buffer size for the internal channel
    pub buffer_size: usize,
    /// Whether to drop old messages when full (vs block)
    pub drop_on_full: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100,
            drop_on_full: true,
        }
    }
}

impl StreamConfig {
    /// Create a new config with the given buffer size
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer_size,
            ..Default::default()
        }
    }

    /// Set drop behavior when buffer is full
    pub fn drop_on_full(mut self, drop: bool) -> Self {
        self.drop_on_full = drop;
        self
    }
}

/// A typed stream that produces values of type T
pub struct Stream<T> {
    rx: cc::Receiver<T>,
    active: Arc<AtomicBool>,
}

impl<T: Send + 'static> Stream<T> {
    /// Create a stream from a receiver
    pub fn from_receiver(rx: Receiver<T>) -> Self {
        Self {
            rx: rx.inner,
            active: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Receive the next value
    pub fn next(&self) -> Option<T> {
        if !self.active.load(Ordering::Relaxed) {
            return None;
        }
        self.rx.recv().ok()
    }

    /// Receive the next value with a timeout
    pub fn next_timeout(&self, timeout: Duration) -> Option<T> {
        if !self.active.load(Ordering::Relaxed) {
            return None;
        }
        self.rx.recv_timeout(timeout).ok()
    }

    /// Try to receive without blocking
    pub fn try_next(&self) -> Option<T> {
        if !self.active.load(Ordering::Relaxed) {
            return None;
        }
        self.rx.try_recv().ok()
    }

    /// Get the latest value, discarding older ones
    pub fn latest(&self) -> Option<T> {
        if !self.active.load(Ordering::Relaxed) {
            return None;
        }
        let mut latest = None;
        while let Ok(v) = self.rx.try_recv() {
            latest = Some(v);
        }
        latest
    }

    /// Stop the stream
    pub fn stop(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Check if the stream is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Create an iterator over stream values
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        std::iter::from_fn(|| self.next())
    }

    /// Map stream values to a new type
    pub fn map<U, F>(self, f: F) -> MappedStream<T, U, F>
    where
        F: Fn(T) -> U,
        U: Send + 'static,
    {
        MappedStream {
            source: self,
            mapper: f,
            _phantom: PhantomData,
        }
    }

    /// Filter stream values
    pub fn filter<F>(self, predicate: F) -> FilteredStream<T, F>
    where
        F: Fn(&T) -> bool,
    {
        FilteredStream {
            source: self,
            predicate,
        }
    }

    /// Batch stream values
    pub fn batch(self, size: usize, timeout: Duration) -> BatchedStream<T> {
        BatchedStream {
            source: self,
            batch_size: size,
            timeout,
        }
    }
}

/// A stream handle for controlling stream lifetime
pub struct StreamHandle {
    active: Arc<AtomicBool>,
}

impl StreamHandle {
    /// Create a new stream handle
    pub fn new() -> (Self, Arc<AtomicBool>) {
        let active = Arc::new(AtomicBool::new(true));
        (
            Self {
                active: active.clone(),
            },
            active,
        )
    }

    /// Stop the stream
    pub fn stop(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Check if the stream is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

impl Default for StreamHandle {
    fn default() -> Self {
        Self::new().0
    }
}

/// A mapped stream that transforms values
pub struct MappedStream<T, U, F>
where
    F: Fn(T) -> U,
{
    source: Stream<T>,
    mapper: F,
    _phantom: PhantomData<U>,
}

impl<T: Send + 'static, U: Send + 'static, F: Fn(T) -> U> MappedStream<T, U, F> {
    /// Get the next transformed value
    pub fn next(&self) -> Option<U> {
        self.source.next().map(&self.mapper)
    }

    /// Get the next transformed value with timeout
    pub fn next_timeout(&self, timeout: Duration) -> Option<U> {
        self.source.next_timeout(timeout).map(&self.mapper)
    }
}

/// A filtered stream that only passes certain values
pub struct FilteredStream<T, F>
where
    F: Fn(&T) -> bool,
{
    source: Stream<T>,
    predicate: F,
}

impl<T: Send + 'static, F: Fn(&T) -> bool> FilteredStream<T, F> {
    /// Get the next value that passes the predicate
    pub fn next(&self) -> Option<T> {
        loop {
            match self.source.next() {
                Some(v) if (self.predicate)(&v) => return Some(v),
                Some(_) => continue,
                None => return None,
            }
        }
    }

    /// Get the next value with timeout
    pub fn next_timeout(&self, timeout: Duration) -> Option<T> {
        let start = std::time::Instant::now();
        loop {
            let remaining = timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return None;
            }
            match self.source.next_timeout(remaining) {
                Some(v) if (self.predicate)(&v) => return Some(v),
                Some(_) => continue,
                None => return None,
            }
        }
    }
}

/// A batched stream that collects values into groups
pub struct BatchedStream<T> {
    source: Stream<T>,
    batch_size: usize,
    timeout: Duration,
}

impl<T: Send + 'static> BatchedStream<T> {
    /// Get the next batch of values
    pub fn next(&self) -> Option<Vec<T>> {
        let mut batch = Vec::with_capacity(self.batch_size);
        let start = std::time::Instant::now();

        while batch.len() < self.batch_size {
            let remaining = self.timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                break;
            }

            match self.source.next_timeout(remaining) {
                Some(v) => batch.push(v),
                None => break,
            }
        }

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
}

/// Create a stream from a function that produces values
pub fn stream_from_fn<T, F>(f: F, config: StreamConfig) -> Stream<T>
where
    T: Send + 'static,
    F: Fn() -> Option<T> + Send + 'static,
{
    let (tx, rx) = cc::bounded(config.buffer_size);
    let active = Arc::new(AtomicBool::new(true));
    let active_clone = active.clone();

    if config.drop_on_full {
        std::thread::spawn(move || {
            while active_clone.load(Ordering::Relaxed) {
                match f() {
                    Some(v) => {
                        let _ = tx.try_send(v);
                    }
                    None => break,
                }
            }
        });
    } else {
        std::thread::spawn(move || {
            while active_clone.load(Ordering::Relaxed) {
                match f() {
                    Some(v) => {
                        let _ = tx.send(v);
                    }
                    None => break,
                }
            }
        });
    }

    Stream { rx, active }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_from_receiver() {
        let (tx, rx) = cc::bounded(10);
        let stream = Stream::from_receiver(super::super::Receiver { inner: rx });

        tx.send(1).unwrap();
        tx.send(2).unwrap();

        assert_eq!(stream.next(), Some(1));
        assert_eq!(stream.next(), Some(2));
    }

    #[test]
    fn test_stream_latest() {
        let (tx, rx) = cc::bounded(10);
        let stream = Stream::from_receiver(super::super::Receiver { inner: rx });

        tx.send(1).unwrap();
        tx.send(2).unwrap();
        tx.send(3).unwrap();

        assert_eq!(stream.latest(), Some(3));
    }

    #[test]
    fn test_stream_stop() {
        let (tx, rx) = cc::bounded(10);
        let stream = Stream::from_receiver(super::super::Receiver { inner: rx });

        tx.send(1).unwrap();
        stream.stop();

        assert!(!stream.is_active());
        assert_eq!(stream.next(), None);
    }

    #[test]
    fn test_mapped_stream() {
        let (tx, rx) = cc::bounded(10);
        let stream = Stream::from_receiver(super::super::Receiver { inner: rx });
        let mapped = stream.map(|x: i32| x * 2);

        tx.send(5).unwrap();
        assert_eq!(mapped.next(), Some(10));
    }

    #[test]
    fn test_filtered_stream() {
        let (tx, rx) = cc::bounded(10);
        let stream = Stream::from_receiver(super::super::Receiver { inner: rx });
        let filtered = stream.filter(|x: &i32| *x % 2 == 0);

        tx.send(1).unwrap();
        tx.send(2).unwrap();
        tx.send(3).unwrap();
        tx.send(4).unwrap();

        assert_eq!(filtered.next(), Some(2));
        assert_eq!(filtered.next(), Some(4));
    }

    #[test]
    fn test_batched_stream() {
        let (tx, rx) = cc::bounded(10);
        let stream = Stream::from_receiver(super::super::Receiver { inner: rx });
        let batched = stream.batch(3, Duration::from_secs(1));

        tx.send(1).unwrap();
        tx.send(2).unwrap();
        tx.send(3).unwrap();

        let batch = batched.next().unwrap();
        assert_eq!(batch, vec![1, 2, 3]);
    }
}
