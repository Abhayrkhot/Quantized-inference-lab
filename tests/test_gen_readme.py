import importlib.util
from pathlib import Path

from qlab import simulate
from qlab.results import write_rows

_spec = importlib.util.spec_from_file_location(
    "gen_readme", Path(__file__).resolve().parent.parent / "scripts" / "gen_readme.py"
)
gen_readme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_readme)

DOC = "intro\n<!-- qlab:begin m0_simulate -->\nold\n<!-- qlab:end m0_simulate -->\noutro\n"


def test_render_fills_block_and_is_idempotent(tmp_path):
    write_rows(tmp_path / "results/box/m0_simulate.csv", simulate.run(shape=(32, 128)))
    (tmp_path / "README.md").write_text(DOC)
    assert gen_readme.main([], root=tmp_path) == 0
    text = (tmp_path / "README.md").read_text()
    assert "int4 g128 asym" in text and "old" not in text
    assert text.startswith("intro\n") and text.endswith("outro\n")
    assert gen_readme.main(["--check"], root=tmp_path) == 0


def test_check_detects_stale_block(tmp_path):
    write_rows(tmp_path / "results/box/m0_simulate.csv", simulate.run(shape=(32, 128)))
    (tmp_path / "README.md").write_text(DOC)
    assert gen_readme.main(["--check"], root=tmp_path) == 1
    assert (tmp_path / "README.md").read_text() == DOC  # --check never writes


def test_missing_results_render_placeholder(tmp_path):
    (tmp_path / "README.md").write_text(DOC)
    gen_readme.main([], root=tmp_path)
    assert "_Not run yet._" in (tmp_path / "README.md").read_text()
