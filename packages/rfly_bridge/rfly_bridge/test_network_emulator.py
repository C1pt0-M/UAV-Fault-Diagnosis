from pathlib import Path

import pytest

from rfly_bridge.network_emulator import LinkEmulator, LinkProfile, load_profile


def test_packaged_network_profiles_are_loadable() -> None:
    config_path = Path(__file__).with_name("configs") / "network_profiles.csv"

    profile = load_profile(str(config_path), "enhanced_5g")

    assert profile.name == "enhanced_5g"
    assert profile.delay_ms == 20.0
    assert profile.loss_rate == pytest.approx(0.005)
    assert profile.bandwidth_kbps == 50_000.0


def test_network_emulator_adds_monitoring_transport_metadata(monkeypatch) -> None:
    profile = LinkProfile("baseline", 10.0, 0.0, 0.0, 20_000.0)
    emulator = LinkEmulator(
        profile,
        listen=("127.0.0.1", 15551),
        forward=("127.0.0.1", 15550),
        queue_limit=8,
    )
    monkeypatch.setattr("rfly_bridge.network_emulator._now_ms", lambda: 123456)

    payload = emulator._encode_payload(b'{"seq":7}', delay_ms=12.5, net_seq=3)

    assert payload == (
        b'{"seq":7,"net_profile":"baseline","net_seq":3,'
        b'"net_tx_ms":123456,"net_delay_ms":12.5}'
    )
