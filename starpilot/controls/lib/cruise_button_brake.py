"""Brake-now alert and regen backstop for cruise-button Bolts (no ACC, no pedal interceptor).

Measured on a 2023 Bolt EUV, 65 -> 55 mph:
  lowering the set speed   ~8 s   (0.56 m/s^2, same in D and L: the stock cruise uses its own mild regen)
  cancelling cruise in L   ~4 s   (1.1 m/s^2, driver lift-off regen; nothing extra in D)
  regen paddle             ~3.5 s (1.3 m/s^2, same in D and L)
Lowering the set speed is the weakest brake the car has. When the planner needs clearly more, this runs
the same steps a driver would: warn, cancel the stock cruise, pull the regen paddle while needed, and
press RES once the gap is rebuilt. openpilot stays engaged throughout; the car controller reads the
published action and sends the matching CAN frames, and the panda only permits them for a short
window after openpilot's own cancel.

Stages: 0 idle, 1 alert, 2 cancel (CANCEL until stock cruise drops), 3 paddle, 4 resume (RES once eased).
"""
from dataclasses import dataclass

from opendbc.car.gm.values import CAR as GM_CAR

CRUISE_BUTTON_BRAKE_CARS = {GM_CAR.CHEVROLET_BOLT_CC_2017, GM_CAR.CHEVROLET_BOLT_CC_2018_2021, GM_CAR.CHEVROLET_BOLT_CC_2022_2023}

SET_SPEED_DECEL_AUTHORITY = 0.56     # m/s^2, measured
PADDLE_DECEL_AUTHORITY = 1.3         # m/s^2, measured

STAGE_IDLE = 0
STAGE_ALERT = 1
STAGE_CANCEL = 2
STAGE_PADDLE = 3
STAGE_RESUME = 4

# Published to the car controller: 0 normal, 1 hold (no button spam, keep steering), 2 send CANCEL,
# 3 feed the regen paddle, 4 send RESUME.
ACTION_NONE = 0
ACTION_HOLD = 1
ACTION_CANCEL = 2
ACTION_PADDLE = 3
ACTION_RESUME = 4

ALERT_MIN_SPEED = 5.0                # m/s
ALERT_ACCEL = -1.0                   # accel command clearly beyond set-speed authority (m/s^2)
ALERT_ACCEL_CONFIRM_S = 0.5
ALERT_MIN_CLOSING = 3.0              # m/s
ALERT_GAP = 8.0                      # m, gap the required-decel estimate aims to keep
ALERT_REQUIRED_DECEL = 0.9           # m/s^2 needed to match the lead by that gap
ALERT_MIN_LEAD_PROB = 0.85           # vision leads only count when the model is confident
ALERT_RELEASE_ACCEL = -0.5
ALERT_RELEASE_DECEL = 0.45
ALERT_RELEASE_S = 1.0

CANCEL_MIN_SPEED = 8.0               # m/s
CANCEL_ALERT_LEAD_S = 0.7            # the alert must have been up this long first
CANCEL_IMMEDIATE_REQUIRED_DECEL = 1.5
CANCEL_SETTLE_S = 0.5                # after cruise drops: still needed -> paddle, eased -> resume
CANCEL_TIMEOUT_S = 1.5               # cruise did not drop: the paddle cancels it as well

PADDLE_MIN_SPEED = 3.0               # m/s
PADDLE_MIN_HOLD_S = 1.0
PADDLE_MAX_HOLD_S = 8.0
PADDLE_RELEASE_S = 0.5

RESUME_SETTLE_S = 1.0                # eased this long before pressing RES
RESUME_TIMEOUT_S = 3.0               # RES presses not taking
RESUME_GIVEUP_S = 12.0               # conditions never right (too slow, gas held): hand back to the driver
RESUME_LOCKOUT_S = 3.0               # no new cancel right after a resume
BACKSTOP_WINDOW_S = 18.0             # stay inside the panda's 20 s window after our cancel
DISENGAGE_EVENT_S = 0.3              # hold the hand-back event so selfdrived cannot miss it
TIMER_EPS = 1e-6


@dataclass
class BrakeOutputs:
  alert: bool = False
  stage: int = STAGE_IDLE
  action: int = ACTION_NONE
  disengage: bool = False


def required_decel_to_match_lead(v_ego, lead_d_rel, lead_v_lead, gap=ALERT_GAP):
  """Constant deceleration needed to be at the lead's speed once the gap has shrunk to `gap`."""
  closing = float(v_ego) - float(lead_v_lead)
  if closing < ALERT_MIN_CLOSING:
    return 0.0
  room = max(float(lead_d_rel) - gap, 1.0)
  return closing * closing / (2.0 * room)


class CruiseButtonBrake:
  """Alert hysteresis and the cancel -> paddle -> resume stage machine. Call update() every frame."""

  def __init__(self):
    self.reset()

  def reset(self):
    self.stage = STAGE_IDLE
    self.alert = False
    self.accel_s = 0.0
    self.release_s = 0.0
    self.alert_s = 0.0
    self.stage_s = 0.0
    self.dropped_s = 0.0
    self.hold_s = 0.0
    self.paddle_release_s = 0.0
    self.settle_s = 0.0
    self.resume_s = 0.0
    self.since_cancel_s = 0.0
    self.lockout_s = 0.0
    self.disengage_s = 0.0

  def _enter(self, stage):
    self.stage = stage
    self.stage_s = 0.0
    self.dropped_s = 0.0
    self.hold_s = 0.0
    self.paddle_release_s = 0.0
    self.settle_s = 0.0
    self.resume_s = 0.0

  def _hand_back(self):
    self.reset()
    self.disengage_s = DISENGAGE_EVENT_S
    return BrakeOutputs(alert=False, stage=STAGE_IDLE, action=ACTION_NONE, disengage=True)

  def update(self, dt, *, enabled, cruise_active, cruise_available, v_ego, accel_cmd, lead_status, lead_d_rel,
             lead_v_lead, lead_prob, gas_pressed, brake_pressed, driver_regen, backstop_allowed, min_resume_speed):
    if not enabled:
      self.reset()
      return BrakeOutputs()

    if self.disengage_s > TIMER_EPS:
      self.disengage_s -= dt
      return BrakeOutputs(disengage=True)

    required = 0.0
    if lead_status and float(lead_prob) >= ALERT_MIN_LEAD_PROB:
      required = required_decel_to_match_lead(v_ego, lead_d_rel, lead_v_lead)
    needed = accel_cmd <= ALERT_ACCEL or required >= ALERT_REQUIRED_DECEL
    eased = accel_cmd > ALERT_RELEASE_ACCEL and required < ALERT_RELEASE_DECEL
    driver_override = gas_pressed or brake_pressed or driver_regen

    if self.stage in (STAGE_IDLE, STAGE_ALERT) and v_ego < ALERT_MIN_SPEED:
      self.reset()
      return BrakeOutputs()

    self.lockout_s = max(self.lockout_s - dt, 0.0)
    if self.stage >= STAGE_CANCEL:
      self.since_cancel_s += dt
    self.stage_s += dt

    # --- alert hysteresis, runs in every stage
    self.accel_s = self.accel_s + dt if accel_cmd <= ALERT_ACCEL else 0.0
    if not self.alert:
      if self.accel_s + TIMER_EPS >= ALERT_ACCEL_CONFIRM_S or required >= ALERT_REQUIRED_DECEL:
        self.alert = True
        self.alert_s = 0.0
        self.release_s = 0.0
    else:
      self.alert_s += dt
      self.release_s = self.release_s + dt if eased else 0.0
      if self.release_s + TIMER_EPS >= ALERT_RELEASE_S:
        self.alert = False
        self.accel_s = 0.0
        self.alert_s = 0.0
    if self.stage == STAGE_IDLE and self.alert:
      self.stage = STAGE_ALERT
    elif self.stage == STAGE_ALERT and not self.alert:
      self.stage = STAGE_IDLE

    action = ACTION_NONE

    if self.stage in (STAGE_IDLE, STAGE_ALERT):
      can_cancel = (backstop_allowed and cruise_active and not driver_override and
                    v_ego >= CANCEL_MIN_SPEED and self.lockout_s <= TIMER_EPS)
      sustained = self.alert and self.alert_s + TIMER_EPS >= CANCEL_ALERT_LEAD_S and needed
      if can_cancel and (sustained or required >= CANCEL_IMMEDIATE_REQUIRED_DECEL):
        self.alert = True
        self._enter(STAGE_CANCEL)
        self.since_cancel_s = 0.0
        action = ACTION_CANCEL

    elif self.stage == STAGE_CANCEL:
      if brake_pressed:
        return self._hand_back()
      if gas_pressed:
        self._enter(STAGE_RESUME)
        action = ACTION_HOLD
      elif cruise_active:
        action = ACTION_CANCEL
        if self.stage_s + TIMER_EPS >= CANCEL_TIMEOUT_S:
          self._enter(STAGE_PADDLE)
          action = ACTION_PADDLE
      else:
        self.dropped_s += dt
        action = ACTION_HOLD
        if self.dropped_s + TIMER_EPS >= CANCEL_SETTLE_S:
          if needed:
            self._enter(STAGE_PADDLE)
            action = ACTION_PADDLE
          elif eased or self.dropped_s >= CANCEL_SETTLE_S + 2.0:
            self._enter(STAGE_RESUME)

    elif self.stage == STAGE_PADDLE:
      if brake_pressed:
        return self._hand_back()
      self.hold_s += dt
      self.paddle_release_s = self.paddle_release_s + dt if eased else 0.0
      done = (gas_pressed or driver_regen or v_ego < PADDLE_MIN_SPEED or
              self.hold_s + TIMER_EPS >= PADDLE_MAX_HOLD_S or
              (self.hold_s + TIMER_EPS >= PADDLE_MIN_HOLD_S and self.paddle_release_s + TIMER_EPS >= PADDLE_RELEASE_S))
      if done:
        self._enter(STAGE_RESUME)
        action = ACTION_HOLD
      else:
        action = ACTION_PADDLE

    elif self.stage == STAGE_RESUME:
      if brake_pressed:
        return self._hand_back()
      if cruise_active:
        # RES took: back to normal set-speed control
        self.lockout_s = RESUME_LOCKOUT_S
        self.stage = STAGE_ALERT if self.alert else STAGE_IDLE
        self.stage_s = 0.0
        return BrakeOutputs(alert=self.alert, stage=self.stage, action=ACTION_NONE)
      if self.since_cancel_s >= BACKSTOP_WINDOW_S:
        return self._hand_back()
      if needed and backstop_allowed and not driver_override and v_ego >= PADDLE_MIN_SPEED:
        self._enter(STAGE_PADDLE)
        action = ACTION_PADDLE
      else:
        ready = (not needed and not gas_pressed and cruise_available and v_ego >= min_resume_speed)
        self.settle_s = self.settle_s + dt if ready else 0.0
        if self.settle_s + TIMER_EPS >= RESUME_SETTLE_S:
          action = ACTION_RESUME
          self.resume_s += dt
        else:
          action = ACTION_HOLD
        if self.resume_s + TIMER_EPS >= RESUME_TIMEOUT_S or self.stage_s + TIMER_EPS >= RESUME_GIVEUP_S:
          return self._hand_back()

    return BrakeOutputs(alert=self.alert, stage=self.stage, action=action)
