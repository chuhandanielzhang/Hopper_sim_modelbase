"""
Pure-Python **fake LCM** transport (for MuJoCo ↔ controller wiring).

Why this exists:
- The real robot code under `Hopper-aero/` uses MIT LCM (`import lcm`) + generated `*_lcmt` message classes.
- This workspace environment may not have the real `lcm` Python bindings installed.
- For simulation, we only need a *small* subset of the API to make the IO pipeline look like hardware.

What this implements (subset of the `lcm` Python API):
- `lcm.LCM(url)`
- `LCM.subscribe(channel, callback)`
- `LCM.publish(channel, bytes_payload)`
- `LCM.handle()`        (blocking)
- `LCM.handle_timeout(ms)` (non-blocking / timed)

Transport:
- UDP multicast using an `udpm://<ip>:<port>?ttl=<N>` URL (same shape as real LCM),
  but **NOT wire-compatible** with real LCM. Both sender and receiver must use this module.

Packet framing (very small, robust enough for our use):
- uint16_be: channel name length
- bytes: channel name (utf-8)
- bytes: raw payload
"""

from __future__ import annotations

import select
import socket
import struct
from typing import Callable
from urllib.parse import parse_qs, urlparse


class EventLog:  # pragma: no cover
    """Placeholder for the real LCM EventLog API (not implemented here)."""

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Fake lcm.EventLog is not implemented. "
            "Install the real LCM bindings if you need log read/write."
        )


class LCM:
    def __init__(self, url: str | None = None):
        if url is None:
            url = "udpm://239.255.76.67:7667?ttl=255"

        u = urlparse(url)
        if u.scheme not in ("udpm", "udp"):
            raise ValueError(f"Unsupported LCM URL scheme {u.scheme!r}. Expected 'udpm://'.")

        mcast_ip = u.hostname or "239.255.76.67"
        port = int(u.port or 7667)
        qs = parse_qs(u.query)
        ttl = int(qs.get("ttl", ["1"])[0])
        ttl = max(0, min(255, ttl))

        self.url = str(url)
        self.mcast_ip = str(mcast_ip)
        self.port = int(port)
        self.ttl = int(ttl)

        self._subs: dict[str, list[Callable[[str, bytes], None]]] = {}

        # RX socket (join multicast group)
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            rx.bind(("", self.port))
        except OSError:
            rx.bind(("0.0.0.0", self.port))

        mreq = struct.pack("4s4s", socket.inet_aton(self.mcast_ip), socket.inet_aton("0.0.0.0"))
        rx.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        rx.setblocking(True)
        self._rx = rx

        # TX socket
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        # IP_MULTICAST_TTL expects an unsigned byte on most systems.
        tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, struct.pack("B", self.ttl))
        tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self._tx = tx

    def subscribe(self, channel: str, callback: Callable[[str, bytes], None]):
        channel = str(channel)
        if not channel:
            raise ValueError("channel must be non-empty")
        self._subs.setdefault(channel, []).append(callback)
        # Real LCM returns a Subscription object; for our uses, returning the pair is enough.
        return (channel, callback)

    def publish(self, channel: str, data: bytes):
        channel_b = str(channel).encode("utf-8")
        if len(channel_b) > 65535:
            raise ValueError("channel name too long")
        payload = bytes(data)
        pkt = struct.pack(">H", len(channel_b)) + channel_b + payload
        self._tx.sendto(pkt, (self.mcast_ip, self.port))

    def _dispatch_one(self, pkt: bytes) -> bool:
        if len(pkt) < 2:
            return False
        (n,) = struct.unpack(">H", pkt[0:2])
        if len(pkt) < 2 + n:
            return False
        try:
            chan = pkt[2 : 2 + n].decode("utf-8")
        except Exception:
            return False
        payload = pkt[2 + n :]
        cbs = self._subs.get(chan, [])
        for cb in list(cbs):
            cb(chan, payload)
        return True

    def handle_timeout(self, timeout_ms: int) -> int:
        timeout_s = max(0.0, float(timeout_ms) / 1000.0)
        r, _, _ = select.select([self._rx], [], [], timeout_s)
        if not r:
            return 0
        pkt, _addr = self._rx.recvfrom(65535)
        return 1 if self._dispatch_one(pkt) else 0

    def handle(self) -> int:
        pkt, _addr = self._rx.recvfrom(65535)
        return 1 if self._dispatch_one(pkt) else 0

    def close(self):
        try:
            self._rx.close()
        except Exception:
            pass
        try:
            self._tx.close()
        except Exception:
            pass


