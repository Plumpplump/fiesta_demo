# Author: David Li, SISHK
"""Integration smoke test: the real app mounts its 60/40 layout.

The ``flet_app`` fixture comes from the flet pytest plugin; never define your
own. Marked ``integration`` because it drives a provisioned Flutter test host -
skipped unless run via ``flet test`` or ``FLET_RUN_INTEGRATION=1`` (see
``conftest.py``).
"""

import flet.testing as ftt
import pytest

pytestmark = pytest.mark.integration


async def test_app_mounts_camera_and_status_panes(flet_app: ftt.FletTestApp):
    tester = flet_app.tester
    await tester.pump_and_settle()

    assert (await tester.find_by_key("preview_box")).count == 1
    assert (await tester.find_by_key("status_list")).count == 1
    assert (await tester.find_by_text("Who is in view")).count == 1