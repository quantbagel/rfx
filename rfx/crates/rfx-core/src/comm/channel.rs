//! Lock-free channels for inter-thread communication
//!
//! Wrapper around crossbeam-channel with pi-specific ergonomics.

use crossbeam_channel::{self as cc, RecvTimeoutError, TryRecvError, TrySendError};
use std::time::Duration;

use crate::{Error, Result};

/// Sender half of a channel
#[derive(Debug)]
pub struct Sender<T> {
    inner: cc::Sender<T>,
}

impl<T> Sender<T> {
    /// Send a value, blocking until space is available
    #[inline]
    pub fn send(&self, value: T) -> Result<()> {
        self.inner.send(value).map_err(|_| Error::ChannelClosed)
    }

    /// Try to send without blocking
    #[inline]
    pub fn try_send(&self, value: T) -> Result<()> {
        match self.inner.try_send(value) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(Error::ChannelFull),
            Err(TrySendError::Disconnected(_)) => Err(Error::ChannelClosed),
        }
    }

    /// Check if the channel is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Check if the channel is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    /// Get the number of messages in the channel
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Get the channel capacity (None for unbounded)
    #[inline]
    pub fn capacity(&self) -> Option<usize> {
        self.inner.capacity()
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Receiver half of a channel
#[derive(Debug)]
pub struct Receiver<T> {
    pub(crate) inner: cc::Receiver<T>,
}

impl<T> Receiver<T> {
    /// Receive a value, blocking until one is available
    #[inline]
    pub fn recv(&self) -> Result<T> {
        self.inner.recv().map_err(|_| Error::ChannelClosed)
    }

    /// Try to receive without blocking
    #[inline]
    pub fn try_recv(&self) -> Result<Option<T>> {
        match self.inner.try_recv() {
            Ok(v) => Ok(Some(v)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(Error::ChannelClosed),
        }
    }

    /// Receive with a timeout
    #[inline]
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<T>> {
        match self.inner.recv_timeout(timeout) {
            Ok(v) => Ok(Some(v)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(Error::ChannelClosed),
        }
    }

    /// Drain all available messages
    #[inline]
    pub fn drain(&self) -> Vec<T> {
        let mut v = Vec::with_capacity(self.inner.len());
        while let Ok(msg) = self.inner.try_recv() {
            v.push(msg);
        }
        v
    }

    /// Get the latest message, discarding older ones
    #[inline]
    pub fn latest(&self) -> Result<Option<T>> {
        let mut latest = match self.inner.try_recv() {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        while let Ok(v) = self.inner.try_recv() {
            latest = v;
        }
        Ok(Some(latest))
    }

    /// Check if the channel is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the number of messages in the channel
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Create an iterator over received messages
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.inner.iter()
    }

    /// Create a try-iterator that doesn't block
    #[inline]
    pub fn try_iter(&self) -> impl Iterator<Item = T> + '_ {
        self.inner.try_iter()
    }
}

impl<T> Clone for Receiver<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> IntoIterator for Receiver<T> {
    type Item = T;
    type IntoIter = cc::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

/// Create a bounded channel with the specified capacity
pub fn bounded_channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    let (tx, rx) = cc::bounded(capacity);
    (Sender { inner: tx }, Receiver { inner: rx })
}

/// Create an unbounded channel
pub fn unbounded_channel<T>() -> (Sender<T>, Receiver<T>) {
    let (tx, rx) = cc::unbounded();
    (Sender { inner: tx }, Receiver { inner: rx })
}

/// A bidirectional channel (useful for request-response patterns)
#[derive(Debug)]
pub struct Channel<Req, Resp> {
    pub request_tx: Sender<Req>,
    pub request_rx: Receiver<Req>,
    pub response_tx: Sender<Resp>,
    pub response_rx: Receiver<Resp>,
}

impl<Req, Resp> Channel<Req, Resp> {
    /// Create a new bidirectional channel
    pub fn new(request_capacity: usize, response_capacity: usize) -> Self {
        let (request_tx, request_rx) = bounded_channel(request_capacity);
        let (response_tx, response_rx) = bounded_channel(response_capacity);
        Self {
            request_tx,
            request_rx,
            response_tx,
            response_rx,
        }
    }

    /// Create a channel split into client and server halves
    pub fn split(
        request_capacity: usize,
        response_capacity: usize,
    ) -> (ChannelClient<Req, Resp>, ChannelServer<Req, Resp>) {
        let (request_tx, request_rx) = bounded_channel(request_capacity);
        let (response_tx, response_rx) = bounded_channel(response_capacity);
        (
            ChannelClient {
                request_tx,
                response_rx,
            },
            ChannelServer {
                request_rx,
                response_tx,
            },
        )
    }
}

/// Client half of a bidirectional channel
#[derive(Debug)]
pub struct ChannelClient<Req, Resp> {
    pub request_tx: Sender<Req>,
    pub response_rx: Receiver<Resp>,
}

impl<Req, Resp> ChannelClient<Req, Resp> {
    /// Send a request
    pub fn send(&self, request: Req) -> Result<()> {
        self.request_tx.send(request)
    }

    /// Receive a response
    pub fn recv(&self) -> Result<Resp> {
        self.response_rx.recv()
    }

    /// Send request and wait for response
    pub fn call(&self, request: Req) -> Result<Resp> {
        self.send(request)?;
        self.recv()
    }

    /// Send request and wait for response with timeout
    pub fn call_timeout(&self, request: Req, timeout: Duration) -> Result<Option<Resp>> {
        self.send(request)?;
        self.response_rx.recv_timeout(timeout)
    }
}

/// Server half of a bidirectional channel
#[derive(Debug)]
pub struct ChannelServer<Req, Resp> {
    pub request_rx: Receiver<Req>,
    pub response_tx: Sender<Resp>,
}

impl<Req, Resp> ChannelServer<Req, Resp> {
    /// Receive a request
    pub fn recv(&self) -> Result<Req> {
        self.request_rx.recv()
    }

    /// Send a response
    pub fn send(&self, response: Resp) -> Result<()> {
        self.response_tx.send(response)
    }

    /// Handle requests with a function
    pub fn handle<F>(&self, handler: F) -> Result<()>
    where
        F: Fn(Req) -> Resp,
    {
        let request = self.recv()?;
        let response = handler(request);
        self.send(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_channel() {
        let (tx, rx) = bounded_channel::<i32>(10);
        tx.send(42).unwrap();
        assert_eq!(rx.recv().unwrap(), 42);
    }

    #[test]
    fn test_unbounded_channel() {
        let (tx, rx) = unbounded_channel::<i32>();
        for i in 0..1000 {
            tx.send(i).unwrap();
        }
        for i in 0..1000 {
            assert_eq!(rx.recv().unwrap(), i);
        }
    }

    #[test]
    fn test_try_recv() {
        let (tx, rx) = bounded_channel::<i32>(10);
        assert!(rx.try_recv().unwrap().is_none());
        tx.send(42).unwrap();
        assert_eq!(rx.try_recv().unwrap(), Some(42));
    }

    #[test]
    fn test_drain() {
        let (tx, rx) = bounded_channel::<i32>(10);
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        tx.send(3).unwrap();
        let msgs = rx.drain();
        assert_eq!(msgs, vec![1, 2, 3]);
    }

    #[test]
    fn test_latest() {
        let (tx, rx) = bounded_channel::<i32>(10);
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        tx.send(3).unwrap();
        assert_eq!(rx.latest().unwrap(), Some(3));
    }

    #[test]
    fn test_channel_client_server() {
        let (client, server) = Channel::<String, String>::split(10, 10);

        std::thread::spawn(move || {
            let req = server.recv().unwrap();
            server.send(format!("Echo: {}", req)).unwrap();
        });

        let response = client.call("Hello".to_string()).unwrap();
        assert_eq!(response, "Echo: Hello");
    }
}
