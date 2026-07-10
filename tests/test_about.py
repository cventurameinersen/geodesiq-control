import importlib

import geodesiq

about_module = importlib.import_module("geodesiq.about")
about = geodesiq.about


def test_about_prints_expected_sections(capsys):
    about()
    captured = capsys.readouterr()

    expected_labels = ["geodesiq: geometric optimal control", "geodesiq Version:", "Numpy Version:", "Scipy Version:",
                       "QuTiP Version:", "Matplotlib Version:", "Python Version:", "Number of CPUs:",
                       "Platform Info:", ]

    for label in expected_labels:
        assert label in captured.out


def test_about_optional_dependency_fallbacks(monkeypatch, capsys):
    real_import_module = about_module.importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name in {"matplotlib"}:
            raise ImportError(name)
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(about_module.importlib, "import_module", fake_import_module)

    about_module.about()
    captured = capsys.readouterr()

    assert "Matplotlib Version: None" in captured.out
