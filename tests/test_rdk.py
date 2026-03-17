from unittest.mock import patch

from ultralytics.utils import checks


def test_check_requirements_returns_false_when_install_disabled_and_package_missing():
    assert checks.check_requirements("definitely-not-a-real-package-ultralytics", install=False) is False


def test_check_rdk_requirements_warns_on_unsupported_non_linux_arm64_platform():
    with patch.object(checks, "LINUX", False), patch.object(checks, "ARM64", False), patch.object(
        checks.LOGGER, "warning"
    ) as warning:
        checks.check_rdk_requirements()

    warning.assert_called_once()
