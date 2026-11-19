"""
Quick checks for the server/hospital message format.

Run:  python -m fed.test_protocol
(also works with pytest if it is installed: pytest fed/)
"""
import socket
import threading

import numpy as np

from fed.protocol import (DONE, HELLO, TRAIN, UPDATE, ProtocolError, decode_body, encode,
                          recv_message, send_message)


def _nn_like_weights(seed=0):
    # same shapes and dtype as models/neural_net.py: W1, b1, W2, b2, W3, b3
    rng = np.random.default_rng(seed)
    shapes = [(2352, 128), (128,), (128, 64), (64,), (64, 1), (1,)]
    return [rng.standard_normal(s).astype(np.float32) for s in shapes]


def _roundtrip(msg_type, fields, weights=None):
    data = encode(msg_type, fields, weights)
    return decode_body(data[8:])        # skip the 8-byte length prefix


def test_weights_come_back_exactly():
    weights = _nn_like_weights()
    t, fields, out = _roundtrip(TRAIN, {"round": 3, "epochs": 1, "lr": 0.01,
                                        "batch_size": 64, "seed": 7}, weights)
    assert t == TRAIN
    assert fields == {"round": 3, "epochs": 1, "lr": 0.01, "batch_size": 64, "seed": 7}
    assert len(out) == len(weights)
    for a, b in zip(out, weights):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.array_equal(a, b)      # exact, not just close


def test_message_without_weights():
    t, fields, out = _roundtrip(HELLO, {"hospital_id": 2, "n_samples": 22499})
    assert t == HELLO and fields == {"hospital_id": 2, "n_samples": 22499} and out is None


def test_missing_field_is_rejected():
    try:
        encode(UPDATE, {"round": 1, "hospital_id": 0})   # no n_samples or loss
    except ProtocolError:
        return
    raise AssertionError("missing fields should raise ProtocolError")


def test_unknown_type_is_rejected():
    try:
        encode("HACK", {})
    except ProtocolError:
        return
    raise AssertionError("unknown type should raise ProtocolError")


def test_garbage_is_rejected():
    for junk in [b"", b"\x00\x00\x00\x05abc", b"\x00\x00\x00\x02{}" + b"extra"]:
        try:
            decode_body(junk)
        except ProtocolError:
            continue
        raise AssertionError(f"garbage {junk!r} should raise ProtocolError")


def test_full_round_over_a_real_socket():
    """A tiny server and hospital on this machine run one HELLO/TRAIN/UPDATE/DONE exchange."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))      # port 0: the OS picks a free port
    listener.listen(1)
    port = listener.getsockname()[1]
    global_weights = _nn_like_weights(seed=1)
    result = {}

    def hospital():
        with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
            send_message(s, HELLO, {"hospital_id": 0, "n_samples": 100})
            _, cfg, w = recv_message(s, expect=TRAIN)
            new_w = [x + 1 for x in w]   # pretend to train
            send_message(s, UPDATE, {"round": cfg["round"], "hospital_id": 0,
                                     "n_samples": 100, "loss": 0.25}, new_w)
            result["done"] = recv_message(s, expect=DONE)[1]

    t = threading.Thread(target=hospital)
    t.start()
    conn, _ = listener.accept()
    with conn:
        conn.settimeout(10)
        _, hello, _ = recv_message(conn, expect=HELLO)
        send_message(conn, TRAIN, {"round": 1, "epochs": 1, "lr": 0.01,
                                   "batch_size": 64, "seed": 0}, global_weights)
        _, upd, w = recv_message(conn, expect=UPDATE)
        send_message(conn, DONE, {"rounds": 1})
    t.join(timeout=10)
    listener.close()

    assert hello == {"hospital_id": 0, "n_samples": 100}
    assert upd["loss"] == 0.25 and upd["round"] == 1
    for a, b in zip(w, global_weights):
        assert np.array_equal(a, b + 1)
    assert result["done"] == {"rounds": 1}


def test_wrong_message_order_is_caught():
    a, b = socket.socketpair()
    with a, b:
        send_message(a, DONE, {"rounds": 1})
        try:
            recv_message(b, expect=UPDATE)
        except ProtocolError:
            return
    raise AssertionError("unexpected message type should raise ProtocolError")


def test_closed_connection_is_reported():
    a, b = socket.socketpair()
    data = encode(HELLO, {"hospital_id": 0, "n_samples": 1})
    a.sendall(data[:5])                  # only part of the message, then hang up
    a.close()
    with b:
        try:
            recv_message(b)
        except ConnectionError:
            return
    raise AssertionError("half a message should raise ConnectionError")


if __name__ == "__main__":
    test_weights_come_back_exactly()
    test_message_without_weights()
    test_missing_field_is_rejected()
    test_unknown_type_is_rejected()
    test_garbage_is_rejected()
    test_full_round_over_a_real_socket()
    test_wrong_message_order_is_caught()
    test_closed_connection_is_reported()
    print("all protocol tests passed")
