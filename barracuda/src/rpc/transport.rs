// SPDX-License-Identifier: AGPL-3.0-or-later

//! Platform-agnostic IPC transport abstraction (ecoBin standard).
//!
//! Abstracts over Unix domain sockets and TCP for cross-platform RPC.

use std::io::{Read, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

#[cfg(unix)]
use std::os::unix::net::UnixStream;

use super::IpcError;

/// Platform-agnostic IPC transport (ecoBin standard).
///
/// Abstracts over Unix domain sockets and TCP for cross-platform RPC.
/// Use [`super::resolve_transport`] to obtain a transport from environment or biomeOS discovery.
#[derive(Debug, Clone)]
pub enum Transport {
    /// Unix domain socket (Unix/macOS only).
    #[cfg(unix)]
    Unix(PathBuf),
    /// TCP socket (all platforms).
    Tcp(SocketAddr),
}

/// Stream type that abstracts over Unix and TCP transports.
///
/// Returned by [`connect_transport`] for use with [`super::send_to`] or custom RPC logic.
#[derive(Debug)]
pub enum TransportStream {
    /// Unix domain socket stream (Unix/macOS only).
    #[cfg(unix)]
    Unix(UnixStream),
    /// TCP stream (all platforms).
    Tcp(std::net::TcpStream),
}

impl Read for TransportStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            #[cfg(unix)]
            Self::Unix(s) => s.read(buf),
            Self::Tcp(s) => s.read(buf),
        }
    }
}

impl Write for TransportStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            #[cfg(unix)]
            Self::Unix(s) => s.write(buf),
            Self::Tcp(s) => s.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            #[cfg(unix)]
            Self::Unix(s) => s.flush(),
            Self::Tcp(s) => s.flush(),
        }
    }
}

impl TransportStream {
    pub(super) fn set_timeouts(
        &self,
        transport: &Transport,
        timeout: Option<Duration>,
    ) -> Result<(), IpcError> {
        match (self, transport) {
            #[cfg(unix)]
            (Self::Unix(s), Transport::Unix(socket)) => {
                s.set_read_timeout(timeout)
                    .map_err(|e| IpcError::ConnectionFailed {
                        socket: socket.clone(),
                        source: e,
                    })?;
                s.set_write_timeout(timeout)
                    .map_err(|e| IpcError::ConnectionFailed {
                        socket: socket.clone(),
                        source: e,
                    })?;
            }
            (Self::Tcp(s), Transport::Tcp(addr)) => {
                s.set_read_timeout(timeout)
                    .map_err(|e| IpcError::ConnectionFailedTcp {
                        addr: *addr,
                        source: e,
                    })?;
                s.set_write_timeout(timeout)
                    .map_err(|e| IpcError::ConnectionFailedTcp {
                        addr: *addr,
                        source: e,
                    })?;
            }
            #[cfg(unix)]
            _ => {}
        }
        Ok(())
    }
}

/// Connects to the given transport and returns a stream ready for RPC.
///
/// # Errors
///
/// Returns `Err(IpcError)` if connection fails.
pub fn connect_transport(transport: &Transport) -> Result<TransportStream, IpcError> {
    match transport {
        #[cfg(unix)]
        Transport::Unix(path) => {
            let stream = UnixStream::connect(path).map_err(|e| IpcError::ConnectionFailed {
                socket: path.clone(),
                source: e,
            })?;
            Ok(TransportStream::Unix(stream))
        }
        Transport::Tcp(addr) => {
            let stream =
                std::net::TcpStream::connect(addr).map_err(|e| IpcError::ConnectionFailedTcp {
                    addr: *addr,
                    source: e,
                })?;
            Ok(TransportStream::Tcp(stream))
        }
    }
}
