from stagekit import validate
from stagekit.probe import package_versions, report


def test_package_versions() -> None:
    versions = package_versions("jsonschema", "surely-not-installed-pkg")
    assert versions["surely-not-installed-pkg"] == "missing"
    assert versions["jsonschema"][0].isdigit()


def test_report_without_gpu_is_valid_probe() -> None:
    info = {
        "schema": "stage-probe/v1",
        "stage": "mix",
        **report("0.1.0", ("jsonschema",), gpu=False),
    }
    validate(info, "stage-probe/v1")
