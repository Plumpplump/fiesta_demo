import os

import pytest


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