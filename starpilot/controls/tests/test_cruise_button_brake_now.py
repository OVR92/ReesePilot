import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.starpilot.controls.lib.starpilot_events import (
  CRUISE_BUTTON_BRAKE_NOW_ACCEL_CONFIRM_S,
  CRUISE_BUTTON_BRAKE_NOW_RELEASE_S,
  CruiseButtonBrakeNow,
  required_decel_to_match_lead,
)


def _run(alert, frames, **kw):
  args = dict(enabled=True, v_ego=29.0, accel_cmd=0.0, lead_status=False, lead_d_rel=0.0, lead_v_lead=0.0)
  args.update(kw)
  out = False
  for _ in range(frames):
    out = alert.update(**args)
  return out


def test_required_decel_uses_closing_speed_and_remaining_room():
  # 29 m/s ego, stopped lead 100 m away: 29^2 / (2 * 92) ~= 4.6 m/s^2
  assert required_decel_to_match_lead(29.0, 100.0, 0.0) == pytest.approx(29.0 ** 2 / (2.0 * 92.0))
  # Faster lead or barely closing: nothing required
  assert required_decel_to_match_lead(29.0, 40.0, 31.0) == 0.0
  assert required_decel_to_match_lead(29.0, 40.0, 27.0) == 0.0


def test_brake_now_triggers_on_sustained_hard_accel_request():
  alert = CruiseButtonBrakeNow()
  confirm = int(round(CRUISE_BUTTON_BRAKE_NOW_ACCEL_CONFIRM_S / DT_MDL))
  assert not _run(alert, confirm - 1, accel_cmd=-2.0)
  assert _run(alert, 1, accel_cmd=-2.0)


def test_brake_now_triggers_immediately_on_stopped_lead_needing_more_than_coast():
  alert = CruiseButtonBrakeNow()
  assert _run(alert, 1, lead_status=True, lead_d_rel=110.0, lead_v_lead=0.0)


def test_brake_now_ignores_faster_or_slowly_closing_leads():
  alert = CruiseButtonBrakeNow()
  assert not _run(alert, 40, lead_status=True, lead_d_rel=30.0, lead_v_lead=30.0)
  assert not _run(alert, 40, lead_status=True, lead_d_rel=60.0, lead_v_lead=27.5)


def test_brake_now_holds_until_request_eases_for_release_time():
  alert = CruiseButtonBrakeNow()
  assert _run(alert, 1, lead_status=True, lead_d_rel=110.0, lead_v_lead=0.0)
  release = int(round(CRUISE_BUTTON_BRAKE_NOW_RELEASE_S / DT_MDL))
  assert _run(alert, release - 1, accel_cmd=-0.2)
  assert not _run(alert, 1, accel_cmd=-0.2)


def test_brake_now_is_inert_when_disabled_or_slow():
  alert = CruiseButtonBrakeNow()
  assert not _run(alert, 40, enabled=False, accel_cmd=-3.0)
  assert not _run(alert, 40, v_ego=3.0, accel_cmd=-3.0, lead_status=True, lead_d_rel=10.0, lead_v_lead=0.0)
