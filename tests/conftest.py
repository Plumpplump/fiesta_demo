# Author: David Li, SISHK
import os

import pytest

# The integration module imports `flet.testing`, which needs the `flet[test]`
# extra (scikit-image and friends). Without it, plain `pytest` must still be a
# clean fast run instead of a collection error.
collect_ignore: list[str] = []
try:  # pragma: no cover - depends on the local environment
    import flet.testing  # noqa: F401
except Exception:  # pragma: no cover
    collect_ignore.append("test_main.py")


def pytest_collection_modifyitems(config, items):
    """Integration tests need a provisioned Flutter test host.

    They run under ``flet test`` (which sets FLET_TEST_FLUTTER_APP_DIR) or when
    FLET_RUN_INTEGRATION=1 is set explicitly; a plain ``pytest`` run skips them
    so the fast suite stays free of the Flutter toolchain.
    """
    if os.environ.get("FLET_RUN_INTEGRATION") == "1":
        return
    if os.environ.get("FLET_TEST_FLUTTER_APP_DIR"):
        return
    skip = pytest.mark.skip(
        reason="integration test: run via `flet test` or FLET_RUN_INTEGRATION=1"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)