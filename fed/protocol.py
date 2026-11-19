"""
How the server and the hospitals talk to each other over the network.

Plain TCP sockets from Python's standard library, no extra packages.

One FedAvg round:
    hospital -> server   HELLO     {"hospital_id", "n_samples"}            (once, on connect)
    server   -> hospital TRAIN     {"round", "epochs", "lr", "batch_size", "seed"} + global weights
    hospital -> server   UPDATE    {"round", "hospital_id", "n_samples", "loss"} + new weights
    ...repeat TRAIN / UPDATE for every round...
    server   -> hospital DONE      {"rounds"}                               (training finished)

Only weights and a few numbers ever leave a hospital, never images or labels.

What one message looks like on the wire:

    [8 bytes: body length][4 bytes: header length][header: JSON text][weights: .npz bytes]

- The 8-byte length comes first because TCP is a stream, not separate messages: the
  receiver must know how many bytes belong to this message before it can read it.
- The header is JSON: the message type plus small values (round number, sizes, ...).
- The weights are the list of numpy arrays from model.get_weights(), stored in numpy's
  own .npz format. That keeps exact dtypes and shapes, so no precision is lost.
  Loading uses allow_pickle=False, so a bad message can never run code on the receiver.
"""
import io
import json
import socket
import struct

import numpy as np

# message types
HELLO = "HELLO"
TRAIN = "TRAIN"
UPDATE = "UPDATE"
DONE = "DONE"
MESSAGE_TYPES = (HELLO, TRAIN, UPDATE, DONE)

# fields each message type must carry in its header (beyond "type")
REQUIRED_FIELDS = {
    HELLO: ("hospital_id", "n_samples"),
    TRAIN: ("round", "epochs", "lr", "batch_size", "seed"),
    UPDATE: ("round", "hospital_id", "n_samples", "loss"),
    DONE: ("rounds",),
}

DEFAULT_PORT = 5000
MAX_MESSAGE_BYTES = 200 * 1024 * 1024   # 200 MB; the NN's weights are about 1.2 MB

_BODY_LEN = struct.Struct("!Q")     # 8-byte unsigned int, network byte order
_HEADER_LEN = struct.Struct("!I")   # 4-byte unsigned int


class ProtocolError(Exception):
    """The other side sent something that does not follow this protocol."""


# ---------------------------------------------------------------------------
# Turning a message into bytes and back (no networking here, easy to test)
# ---------------------------------------------------------------------------
def encode(msg_type, fields=None, weights=None):
    """Message -> bytes ready to send (including the 8-byte length prefix).

    msg_type: one of MESSAGE_TYPES
    fields:   dict of small JSON-friendly values (ints, floats, strings)
    weights:  list of numpy arrays, or None for messages without weights
    """
    fields = dict(fields or {})
    _check_fields(msg_type, fields)

    header = dict(fields, type=msg_type, n_arrays=0 if weights is None else len(weights))
    header_bytes = json.dumps(header).encode("utf-8")

    weight_bytes = b""
    if weights is not None:
        buf = io.BytesIO()
        # arr_0, arr_1, ... keep the order of get_weights()
        np.savez(buf, *[np.asarray(w) for w in weights])
        weight_bytes = buf.getvalue()

    body = _HEADER_LEN.pack(len(header_bytes)) + header_bytes + weight_bytes
    if len(body) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message is {len(body)} bytes, over the {MAX_MESSAGE_BYTES} limit")
    return _BODY_LEN.pack(len(body)) + body


def decode_body(body):
    """Bytes (without the 8-byte length prefix) -> (msg_type, fields, weights).
    weights is a list of numpy arrays, or None if the message had none."""
    if len(body) < _HEADER_LEN.size:
        raise ProtocolError("message too short")
    (header_len,) = _HEADER_LEN.unpack_from(body)
    start = _HEADER_LEN.size
    if start + header_len > len(body):
        raise ProtocolError("header length is larger than the message")
    try:
        header = json.loads(body[start:start + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"header is not valid JSON: {e}") from None
    if not isinstance(header, dict):
        raise ProtocolError("header must be a JSON object")

    msg_type = header.pop("type", None)
    n_arrays = header.pop("n_arrays", 0)
    _check_fields(msg_type, header)

    weights = None
    rest = body[start + header_len:]
    if n_arrays:
        try:
            with np.load(io.BytesIO(rest), allow_pickle=False) as npz:
                weights = [npz[f"arr_{i}"] for i in range(n_arrays)]
        except Exception as e:  # bad zip data, missing array, pickled object, ...
            raise ProtocolError(f"could not read weights: {e}") from None
    elif rest:
        raise ProtocolError("unexpected bytes after the header")
    return msg_type, header, weights


def _check_fields(msg_type, fields):
    if msg_type not in MESSAGE_TYPES:
        raise ProtocolError(f"unknown message type: {msg_type!r}")
    missing = [f for f in REQUIRED_FIELDS[msg_type] if f not in fields]
    if missing:
        raise ProtocolError(f"{msg_type} message is missing {missing}")


# ---------------------------------------------------------------------------
# Sending and receiving over a socket
# ---------------------------------------------------------------------------
def send_message(sock, msg_type, fields=None, weights=None):
    """Send one message. sendall() keeps sending until every byte is out."""
    sock.sendall(encode(msg_type, fields, weights))


def recv_message(sock, expect=None):
    """Wait for one whole message and return (msg_type, fields, weights).

    expect: a message type (or tuple of types); if given, any other type raises
            ProtocolError, which catches the two sides getting out of step.
    """
    (body_len,) = _BODY_LEN.unpack(_recv_exact(sock, _BODY_LEN.size))
    if body_len > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"incoming message says {body_len} bytes, over the limit")
    msg_type, fields, weights = decode_body(_recv_exact(sock, body_len))

    if expect is not None:
        allowed = (expect,) if isinstance(expect, str) else tuple(expect)
        if msg_type not in allowed:
            raise ProtocolError(f"expected {allowed}, got {msg_type}")
    return msg_type, fields, weights


def _recv_exact(sock, n):
    """recv() may return fewer bytes than asked for, so keep reading until we have n.
    An empty read means the other side closed the connection."""
    chunks, got = [], 0
    while got < n:
        chunk = sock.recv(min(n - got, 1 << 20))
        if not chunk:
            raise ConnectionError(f"connection closed after {got} of {n} bytes")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def weights_nbytes(weights):
    """Size of the raw weight arrays in bytes (for the 'size sent per round' table)."""
    return int(sum(np.asarray(w).nbytes for w in weights))
