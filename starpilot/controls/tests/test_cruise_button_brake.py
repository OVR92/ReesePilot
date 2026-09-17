import pytest

from openpilot.starpilot.controls.lib.cruise_button_brake import (
  ACTION_CANCEL, ACTION_HOLD, ACTION_NONE, ACTION_PADDLE, ACTION_RESUME,
  ALERT_ACCEL_CONFIRM_S, ALERT_RELEASE_S, CANCEL_ALERT_LEAD_S, CANCEL_SETTLE_S, PADDLE_MAX_HOLD_S, PADDLE_MIN_HOLD_S,
  PADDLE_RELEASE_S, RESUME_GIVEUP_S, RESUME_SETTLE_S, RESUME_TIMEOUT_S,
  STAGE_ALERT, STAGE_CANCEL, STAGE_IDLE, STAGE_PADDLE, STAGE_RESUME,
  CruiseButtonBrake, required_decel_to_match_lead,
)

DT = 0.05
STOPPED_LEAD = dict(lead_status=True, lead_d_rel=110.0, lead_v_lead=0.0)      # needs ~4.1 m/s^2
MODERATE_LEAD = dict(lead_status=True, lead_d_rel=34.0, lead_v_lead=22.0)    # needs ~0.94 m/s^2


def _run(brake, seconds, **kw):
  args = dict(enabled=True, cruise_active=True, cruise_available=True, v_ego=29.0, accel_cmd=0.0, lead_status=False,
              lead_d_rel=0.0, lead_v_lead=0.0, lead_prob=1.0, gas_pressed=False, brake_pressed=False, driver_regen=False,
              backstop_allowed=True, min_resume_speed=11.2)
  args.update(kw)
  out = None
  for _ in range(max(1, int(round(seconds / DT)))):
    out = brake.update(DT, **args)
  return out


def _drive_to_paddle(brake):
  out = _run(brake, DT, **STOPPED_LEAD)
  assert (out.stage, out.action) == (STAGE_CANCEL, ACTION_CANCEL)
  out = _run(brake, CANCEL_SETTLE_S, cruise_active=False, **STOPPED_LEAD)
  assert (out.stage, out.action) == (STAGE_PADDLE, ACTION_PADDLE)
  return out


def test_required_decel_uses_closing_speed_and_remaining_room():
  assert required_decel_to_match_lead(29.0, 100.0, 0.0) == pytest.approx(29.0 ** 2 / (2.0 * 92.0))
  assert required_decel_to_match_lead(29.0, 40.0, 31.0) == 0.0
  assert required_decel_to_match_lead(29.0, 40.0, 27.0) == 0.0


def test_alert_then_cancel_on_sustained_hard_accel_request():
  brake = CruiseButtonBrake()
  out = _run(brake, ALERT_ACCEL_CONFIRM_S - DT, accel_cmd=-1.5)
  assert (out.alert, out.stage, out.action) == (False, STAGE_IDLE, ACTION_NONE)
  out = _run(brake, DT, accel_cmd=-1.5)
  assert (out.alert, out.stage, out.action) == (True, STAGE_ALERT, ACTION_NONE)
  out = _run(brake, CANCEL_ALERT_LEAD_S - DT, accel_cmd=-1.5)
  assert (out.stage, out.action) == (STAGE_ALERT, ACTION_NONE)
  out = _run(brake, DT, accel_cmd=-1.5)
  assert (out.alert, out.stage, out.action) == (True, STAGE_CANCEL, ACTION_CANCEL)


def test_stopped_lead_cancels_immediately_then_paddles_then_resumes():
  brake = CruiseButtonBrake()
  _drive_to_paddle(brake)
  # still needed: paddle stays on past the minimum hold
  out = _run(brake, PADDLE_MIN_HOLD_S, cruise_active=False, **STOPPED_LEAD)
  assert (out.stage, out.action) == (STAGE_PADDLE, ACTION_PADDLE)
  # lead gone / gap rebuilt: paddle releases after PADDLE_RELEASE_S
  out = _run(brake, PADDLE_RELEASE_S, cruise_active=False)
  assert (out.stage, out.action) == (STAGE_RESUME, ACTION_HOLD)
  # eased for RESUME_SETTLE_S -> press RES
  out = _run(brake, RESUME_SETTLE_S - DT, cruise_active=False)
  assert out.action == ACTION_HOLD
  out = _run(brake, DT, cruise_active=False)
  assert out.action == ACTION_RESUME
  # stock cruise back: normal operation, no disengage
  out = _run(brake, DT, cruise_active=True)
  assert (out.stage, out.action, out.disengage) == (STAGE_IDLE, ACTION_NONE, False)


def test_cancel_settles_to_resume_when_lead_eases_without_paddle():
  brake = CruiseButtonBrake()
  out = _run(brake, DT, **STOPPED_LEAD)
  assert out.action == ACTION_CANCEL
  out = _run(brake, CANCEL_SETTLE_S, cruise_active=False)   # eased right after the cancel
  assert (out.stage, out.action) == (STAGE_RESUME, ACTION_HOLD)


def test_cancel_times_out_into_paddle_if_cruise_does_not_drop():
  brake = CruiseButtonBrake()
  _run(brake, DT, **STOPPED_LEAD)
  out = _run(brake, 1.5, cruise_active=True, **STOPPED_LEAD)
  assert (out.stage, out.action) == (STAGE_PADDLE, ACTION_PADDLE)


def test_paddle_max_hold_then_resume():
  brake = CruiseButtonBrake()
  _drive_to_paddle(brake)
  out = _run(brake, PADDLE_MAX_HOLD_S, cruise_active=False, **STOPPED_LEAD)
  assert out.stage == STAGE_RESUME


def test_brake_press_hands_back_to_driver():
  brake = CruiseButtonBrake()
  _drive_to_paddle(brake)
  out = _run(brake, DT, cruise_active=False, brake_pressed=True, **STOPPED_LEAD)
  assert (out.disengage, out.stage, out.action) == (True, STAGE_IDLE, ACTION_NONE)


def test_resume_gives_up_and_disengages_when_res_does_not_take():
  brake = CruiseButtonBrake()
  _drive_to_paddle(brake)
  out = _run(brake, PADDLE_MIN_HOLD_S, cruise_active=False)          # eased: paddle releases at the minimum hold
  assert (out.stage, out.action) == (STAGE_RESUME, ACTION_HOLD)
  out = _run(brake, RESUME_SETTLE_S, cruise_active=False)            # settled: first RES press
  assert out.action == ACTION_RESUME
  out = _run(brake, RESUME_TIMEOUT_S - 2 * DT, cruise_active=False)
  assert (out.action, out.disengage) == (ACTION_RESUME, False)
  out = _run(brake, DT, cruise_active=False)
  assert out.disengage
  # hand-back is held long enough for selfdrived to see it, then the machine is idle
  assert _run(brake, 0.2, cruise_active=False).disengage
  assert not _run(brake, 0.2, cruise_active=False).disengage


def test_resume_waits_while_too_slow_then_hands_back():
  brake = CruiseButtonBrake()
  _drive_to_paddle(brake)
  out = _run(brake, PADDLE_MIN_HOLD_S, cruise_active=False, v_ego=9.0)
  assert (out.stage, out.action) == (STAGE_RESUME, ACTION_HOLD)
  out = _run(brake, 5.0, cruise_active=False, v_ego=9.0)
  assert (out.stage, out.action, out.disengage) == (STAGE_RESUME, ACTION_HOLD, False)
  out = _run(brake, RESUME_GIVEUP_S - 5.0, cruise_active=False, v_ego=9.0)
  assert out.disengage


def test_needed_again_during_resume_goes_back_to_paddle():
  brake = CruiseButtonBrake()
  _drive_to_paddle(brake)
  _run(brake, PADDLE_MIN_HOLD_S + PADDLE_RELEASE_S, cruise_active=False)
  out = _run(brake, DT, cruise_active=False, **STOPPED_LEAD)
  assert (out.stage, out.action) == (STAGE_PADDLE, ACTION_PADDLE)


def test_no_cancel_without_toggle_driver_input_or_active_cruise():
  for kw in (dict(backstop_allowed=False), dict(gas_pressed=True), dict(driver_regen=True), dict(cruise_active=False)):
    brake = CruiseButtonBrake()
    out = _run(brake, 3.0, **STOPPED_LEAD, **kw)
    assert out.alert and out.stage == STAGE_ALERT and out.action == ACTION_NONE, kw


def test_moderate_lead_alerts_first_and_cancels_only_after_lead_time():
  brake = CruiseButtonBrake()
  out = _run(brake, DT, **MODERATE_LEAD)
  assert (out.alert, out.stage) == (True, STAGE_ALERT)
  out = _run(brake, CANCEL_ALERT_LEAD_S - DT, **MODERATE_LEAD)
  assert out.stage == STAGE_ALERT
  out = _run(brake, DT, **MODERATE_LEAD)
  assert (out.stage, out.action) == (STAGE_CANCEL, ACTION_CANCEL)


def test_ignores_faster_slowly_closing_or_uncertain_leads():
  brake = CruiseButtonBrake()
  assert not _run(brake, 2.0, lead_status=True, lead_d_rel=30.0, lead_v_lead=30.0).alert
  assert not _run(brake, 2.0, lead_status=True, lead_d_rel=60.0, lead_v_lead=27.5).alert
  assert not _run(brake, 2.0, lead_status=True, lead_d_rel=110.0, lead_v_lead=0.0, lead_prob=0.5).alert


def test_alert_releases_after_request_eases():
  brake = CruiseButtonBrake()
  assert _run(brake, DT, backstop_allowed=False, **MODERATE_LEAD).alert
  assert _run(brake, ALERT_RELEASE_S - DT, accel_cmd=-0.2, backstop_allowed=False).alert
  assert not _run(brake, DT, accel_cmd=-0.2, backstop_allowed=False).alert


def test_inert_when_disabled_or_slow():
  brake = CruiseButtonBrake()
  out = _run(brake, 2.0, enabled=False, accel_cmd=-3.0)
  assert (out.alert, out.action) == (False, ACTION_NONE)
  out = _run(brake, 2.0, v_ego=3.0, accel_cmd=-3.0, lead_status=True, lead_d_rel=10.0, lead_v_lead=0.0)
  assert (out.alert, out.action) == (False, ACTION_NONE)
