import math

from qlab import simulate


def test_simulate_rows():
    rows = simulate.run(shape=(64, 256), seed=0)
    per_matrix = 1 + len(simulate.DEFAULT_SPECS)
    assert len(rows) == 2 * per_matrix
    assert {r["matrix"] for r in rows} == {"gaussian", "outliers"}
    fp32 = [r for r in rows if r["format"] == "fp32"]
    assert all(r["bytes"] == 64 * 256 * 4 and r["mse"] == 0 for r in fp32)
    quant = [r for r in rows if r["bits"] != 32]
    assert all(math.isfinite(r["mse"]) and r["bytes"] < 64 * 256 * 4 for r in quant)


def test_outliers_hurt_per_channel_more_than_groups():
    rows = {(r["matrix"], r["format"]): r for r in simulate.run(shape=(256, 1024), seed=0)}
    for bits in (4, 3):
        chan = rows[("outliers", f"int{bits} channel asym")]["mse"]
        group = rows[("outliers", f"int{bits} g128 asym")]["mse"]
        assert group < chan


def test_simulate_is_deterministic():
    a = simulate.run(shape=(32, 128), seed=3)
    b = simulate.run(shape=(32, 128), seed=3)
    assert a == b
