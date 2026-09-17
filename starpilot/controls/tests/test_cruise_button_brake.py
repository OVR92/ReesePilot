import pytest

from openpilot.starpilot.controls.lib.cruise_button_brake import (
  ALERT_ACCEL_CONFIRM_S,
  ALERT_RELEASE_S,
  CANCEL_ALERT_LEAD_S,
  CANCEL_EVENT_S,
  CruiseButtonBrake,
  required_decel_to_match_lead,
)

DT = 0.05


def _run(brake, seconds, **kw):
  args = dict(enabled=True, cruise_active=True, v_ego=29.0, accel_cmd=0.0, lead_status=False, lead_d_rel=0.0,
              lead_v_lead=0.0, lead_prob=1.0, gear_low=True, driver_input=False, cancel_allowed=True)
  args.update(kw)
  out = (False, False)
  for _ in range(int(round(seconds / DT))):
    out = brake.update(DT, **args)
  return out


def test_required_decel_uses_closing_speed_and_remaining_room():
  assert required_decel_to_match_lead(29.0, 100.0, 0.0) == pytest.approx(29.0 ** 2 / (2.0 * 92.0))
  assert required_decel_to_match_lead(29.0, 40.0, 31.0) == 0.0
  assert required_decel_to_match_lead(29.0, 40.0, 27.0) == 0.0


def test_alert_on_sustained_hard_accel_request_then_cancel_after_lead_time():
  brake = CruiseButtonBrake()
  assert _run(brake, ALERT_ACCEL_CONFIRM_S - DT, accel_cmd=-1.5) == (False, False)
  assert _run(brake, DT, accel_cmd=-1.5) == (True, False)
  assert _run(brake, CANCEL_ALERT_LEAD_S - DT, accel_cmd=-1.5) == (True, False)
  assert _run(brake, DT, accel_cmd=-1.5) == (True, True)
  assert brake.cancelled


def test_stopped_lead_needing_far_more_than_coast_cancels_immediately():
  brake = CruiseButtonBrake()
  assert _run(brake, DT, lead_status=True, lead_d_rel=110.0, lead_v_lead=0.0) == (True, True)


def test_moderate_lead_alerts_first_and_cancels_only_after_lead_time():
  brake = CruiseButtonBrake()
  # 29 m/s vs 22 m/s lead at 40 m: 49 / 64 ~= 0.77... use 34 m -> 49/52 ~= 0.94 m/s^2
  kw = dict(lead_status=True, lead_d_rel=34.0, lead_v_lead=22.0)
  assert _run(brake, DT, **kw) == (True, False)
  assert _run(brake, CANCEL_ALERT_LEAD_S - DT, **kw) == (True, False)
  assert _run(brake, DT, **kw) == (True, True)


def test_no_cancel_in_drive_or_with_driver_input_or_toggle_off():
  for kw in (dict(gear_low=False), dict(driver_input=True), dict(cancel_allowed=False), dict(cruise_active=False)):
    brake = CruiseButtonBrake()
    alert, cancel = _run(brake, 3.0, lead_status=True, lead_d_rel=110.0, lead_v_lead=0.0, **kw)
    assert alert and not cancel, kw


def test_ignores_faster_slowly_closing_or_uncertain_leads():
  brake = CruiseButtonBrake()
  assert _run(brake, 2.0, lead_status=True, lead_d_rel=30.0, lead_v_lead=30.0) == (False, False)
  assert _run(brake, 2.0, lead_status=True, lead_d_rel=60.0, lead_v_lead=27.5) == (False, False)
  assert _run(brake, 2.0, lead_status=True, lead_d_rel=110.0, lead_v_lead=0.0, lead_prob=0.5) == (False, False)


def test_alert_releases_after_request_eases():
  brake = CruiseButtonBrake()
  assert _run(brake, DT, lead_status=True, lead_d_rel=34.0, lead_v_lead=22.0, cancel_allowed=False) == (True, False)
  assert _run(brake, ALERT_RELEASE_S - DT, accel_cmd=-0.2) == (True, False)
  assert _run(brake, DT, accel_cmd=-0.2) == (False, False)


def test_cancel_event_is_held_and_alert_persists_after_disengage():
  brake = CruiseButtonBrake()
  kw = dict(lead_status=True, lead_d_rel=110.0, lead_v_lead=0.0)
  assert _run(brake, DT, **kw) == (True, True)
  # openpilot disengages: the cancel event stays up for CANCEL_EVENT_S, the alert while braking is still needed
  assert _run(brake, CANCEL_EVENT_S - 2 * DT, enabled=False, **kw) == (True, True)
  assert _run(brake, 2 * DT, enabled=False, **kw) == (True, False)
  assert _run(brake, 1.0, enabled=False, **kw) == (True, False)
  # lead no longer a threat: alert clears and state resets
  assert _run(brake, DT, enabled=False) == (False, False)
  assert not brake.cancelled


def test_inert_when_disabled_or_slow():
  brake = CruiseButtonBrake()
  assert _run(brake, 2.0, enabled=False, accel_cmd=-3.0) == (False, False)
  assert _run(brake, 2.0, v_ego=3.0, accel_cmd=-3.0, lead_status=True, lead_d_rel=10.0, lead_v_lead=0.0) == (False, False)
