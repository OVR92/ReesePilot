import pytest

from openpilot.starpilot.common import cruise_button_brake_channel as channel
from openpilot.starpilot.controls.lib.cruise_button_brake import (
  ACTION_CANCEL, ACTION_HOLD, ACTION_NONE, ACTION_PADDLE, ACTION_RESUME,
  ATTENTION_RELEASE_S, BRAKE_PADDLE_STALL_S, BRAKE_RELEASE_S, CANCEL_CONFIRM_S, CANCEL_SETTLE_S,
  PADDLE_MAX_HOLD_S, PADDLE_MIN_HOLD_S, PADDLE_RELEASE_S, RESUME_GIVEUP_S, RESUME_SETTLE_S, RESUME_TIMEOUT_S,
  STAGE_ATTENTION, STAGE_CANCEL, STAGE_IDLE, STAGE_PADDLE, STAGE_RESUME,
  CruiseButtonBrake, required_decel_to_match_lead, time_to_contact,
)

DT = 0.05
# 29 m/s ego
STOPPED_LEAD = dict(lead_status=True, lead_d_rel=110.0, lead_v_lead=0.0)     # needs ~4.1 m/s^2, ttc 3.8 s
MILD_LEAD = dict(lead_status=True, lead_d_rel=60.0, lead_v_lead=22.0)        # 49/104 ~= 0.47: attention only
FIRM_LEAD = dict(lead_status=True, lead_d_rel=36.0, lead_v_lead=22.0)        # 49/56 ~= 0.88: backstop
HARD_LEAD = dict(lead_status=True, lead_d_rel=26.0, lead_v_lead=22.0)        # 49/36 ~= 1.36: paddle-level, no red yet


def _run(brake, seconds, **kw):
  args = dict(enabled=True, cruise_active=True, cruise_available=True, v_ego=29.0, accel_cmd=0.0, lead_status=False,
              lead_d_rel=0.0, lead_v_lead=0.0, lead_prob=1.0, gas_pressed=False, brake_pressed=False, driver_regen=False,
              backstop_allowed=True, min_resume_speed=11.2)
  args.update(kw)
  out = None
  for _ in range(max(1, int(round(seconds / DT)))):
    out = brake.update(DT, **args)
  return out


def _drive_to_paddle(brake, lead=STOPPED_LEAD):
  out = _run(brake, DT, **lead)
  assert (out.stage, out.action) == (STAGE_CANCEL, ACTION_CANCEL)
  out = _run(brake, CANCEL_SETTLE_S, cruise_active=False, **lead)
  assert (out.stage, out.action) == (STAGE_PADDLE, ACTION_PADDLE)
  return out


def test_geometry_helpers():
  assert required_decel_to_match_lead(29.0, 100.0, 0.0) == pytest.approx(29.0 ** 2 / (2.0 * 92.0))
  assert required_decel_to_match_lead(29.0, 40.0, 31.0) == 0.0
  assert required_decel_to_match_lead(29.0, 40.0, 27.0) == 0.0
  assert time_to_contact(29.0, 58.0, 0.0) == pytest.approx(2.0)
  assert time_to_contact(29.0, 58.0, 28.0) == float("inf")


def test_planner_request_alone_never_raises_the_red_alert():
  brake = CruiseButtonBrake()
  out = _run(brake, 5.0, accel_cmd=-3.0, backstop_allowed=False)
  assert not out.brake
  assert not out.attention


def test_mild_lead_gets_attention_with_one_chime_and_nothing_else():
  brake = CruiseButtonBrake()
  out = _run(brake, DT, **MILD_LEAD)
  assert (out.attention, out.attention_chime, out.brake, out.stage, out.action) == (True, True, False, STAGE_ATTENTION, ACTION_NONE)
  out = _run(brake, 3.0, **MILD_LEAD)
  assert (out.attention, out.attention_chime, out.brake, out.stage, out.action) == (True, False, False, STAGE_ATTENTION, ACTION_NONE)
  out = _run(brake, ATTENTION_RELEASE_S, lead_status=False)
  assert not out.attention


def test_firm_lead_confirms_then_cancels_without_red_alert():
  brake = CruiseButtonBrake()
  out = _run(brake, CANCEL_CONFIRM_S - DT, **FIRM_LEAD)
  assert (out.attention, out.stage, out.action, out.brake) == (True, STAGE_ATTENTION, ACTION_NONE, False)
  out = _run(brake, DT, **FIRM_LEAD)
  assert (out.stage, out.action, out.brake) == (STAGE_CANCEL, ACTION_CANCEL, False)


def test_hard_lead_cancels_immediately_and_paddles_without_red_alert():
  brake = CruiseButtonBrake()
  out = _drive_to_paddle(brake, HARD_LEAD)
  assert not out.brake


def test_red_alert_when_beyond_the_paddle_or_ttc_short():
  brake = CruiseButtonBrake()
  out = _run(brake, DT, **STOPPED_LEAD)
  assert out.brake and out.stage == STAGE_CANCEL
  brake = CruiseButtonBrake()
  out = _run(brake, DT, lead_status=True, lead_d_rel=8.0, lead_v_lead=25.0, backstop_allowed=False)  # ttc 2.0 s
  assert out.brake
  out = _run(brake, BRAKE_RELEASE_S, lead_status=False, backstop_allowed=False)
  assert not out.brake


def test_red_alert_when_paddle_is_not_gaining():
  brake = CruiseButtonBrake()
  _drive_to_paddle(brake, HARD_LEAD)
  # the stall check reads the hold timer before this frame's increment, so it trips one frame after the threshold
  out = _run(brake, BRAKE_PADDLE_STALL_S, cruise_active=False, **HARD_LEAD)
  assert (out.stage, out.brake) == (STAGE_PADDLE, False)
  out = _run(brake, DT, cruise_active=False, **HARD_LEAD)
  assert (out.stage, out.brake) == (STAGE_PADDLE, True)


def test_stopped_lead_cancels_immediately_then_paddles_then_resumes():
  brake = CruiseButtonBrake()
  _drive_to_paddle(brake)
  out = _run(brake, PADDLE_MIN_HOLD_S, cruise_active=False, **STOPPED_LEAD)
  assert (out.stage, out.action) == (STAGE_PADDLE, ACTION_PADDLE)
  out = _run(brake, PADDLE_RELEASE_S, cruise_active=False)              # eased past the minimum hold: releases
  assert (out.stage, out.action) == (STAGE_RESUME, ACTION_HOLD)
  out = _run(brake, RESUME_SETTLE_S - DT, cruise_active=False)
  assert out.action == ACTION_HOLD
  out = _run(brake, DT, cruise_active=False)
  assert out.action == ACTION_RESUME
  out = _run(brake, DT, cruise_active=True)
  assert (out.stage, out.action, out.disengage) == (STAGE_IDLE, ACTION_NONE, False)


def test_cancel_settles_to_resume_when_lead_eases_without_paddle():
  brake = CruiseButtonBrake()
  out = _run(brake, DT, **STOPPED_LEAD)
  assert out.action == ACTION_CANCEL
  out = _run(brake, CANCEL_SETTLE_S, cruise_active=False)
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
  out = _run(brake, PADDLE_MIN_HOLD_S, cruise_active=False)
  assert (out.stage, out.action) == (STAGE_RESUME, ACTION_HOLD)
  out = _run(brake, RESUME_SETTLE_S, cruise_active=False)
  assert out.action == ACTION_RESUME
  out = _run(brake, RESUME_TIMEOUT_S - 2 * DT, cruise_active=False)
  assert (out.action, out.disengage) == (ACTION_RESUME, False)
  out = _run(brake, DT, cruise_active=False)
  assert out.disengage
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
  _run(brake, PADDLE_MIN_HOLD_S, cruise_active=False)
  out = _run(brake, DT, cruise_active=False, **STOPPED_LEAD)
  assert (out.stage, out.action) == (STAGE_PADDLE, ACTION_PADDLE)


def test_no_cancel_without_toggle_driver_input_or_active_cruise():
  for kw in (dict(backstop_allowed=False), dict(gas_pressed=True), dict(driver_regen=True), dict(cruise_active=False)):
    brake = CruiseButtonBrake()
    out = _run(brake, 3.0, **STOPPED_LEAD, **kw)
    assert out.attention and out.stage == STAGE_ATTENTION and out.action == ACTION_NONE, kw


def test_planner_backup_cancels_only_after_long_sustained_hard_request():
  brake = CruiseButtonBrake()
  out = _run(brake, 1.5 - DT, accel_cmd=-2.0)
  assert out.action == ACTION_NONE
  out = _run(brake, DT, accel_cmd=-2.0)
  assert out.action == ACTION_CANCEL


def test_ignores_faster_slowly_closing_or_uncertain_leads():
  brake = CruiseButtonBrake()
  assert not _run(brake, 2.0, lead_status=True, lead_d_rel=30.0, lead_v_lead=30.0).attention
  assert not _run(brake, 2.0, lead_status=True, lead_d_rel=60.0, lead_v_lead=27.5).attention
  assert not _run(brake, 2.0, lead_status=True, lead_d_rel=110.0, lead_v_lead=0.0, lead_prob=0.5).attention


def test_channel_roundtrip_and_staleness(tmp_path):
  path = str(tmp_path / "action")
  assert channel.read_action(path=path) == 0                       # missing file
  channel.write_action(3, now=100.0, path=path)
  assert channel.read_action(now=100.2, path=path) == 3
  assert channel.read_action(now=100.0 + channel.STALE_S + 0.01, path=path) == 0   # writer stalled
  channel.write_action(0, now=101.0, path=path)
  assert channel.read_action(now=101.1, path=path) == 0
  with open(path, "w", encoding="utf-8") as f:
    f.write("garbage")
  assert channel.read_action(path=path) == 0


def test_inert_when_disabled_or_slow():
  brake = CruiseButtonBrake()
  out = _run(brake, 2.0, enabled=False, **STOPPED_LEAD)
  assert (out.attention, out.brake, out.action) == (False, False, ACTION_NONE)
  out = _run(brake, 2.0, v_ego=3.0, lead_status=True, lead_d_rel=10.0, lead_v_lead=0.0)
  assert (out.attention, out.brake, out.action) == (False, False, ACTION_NONE)
