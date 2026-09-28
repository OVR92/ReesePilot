"""Two-tier warning and regen backstop for cruise-button Bolts (no ACC, no pedal interceptor).

Measured on a 2023 Bolt EUV, 65 -> 55 mph:
  lowering the set speed   ~8 s   (0.56 m/s^2, same in D and L: the stock cruise uses its own mild regen)
  cancelling cruise in L   ~4 s   (1.1 m/s^2, driver lift-off regen; nothing extra in D)
  regen paddle             ~3.5 s (1.3 m/s^2, same in D and L)

Everything here keys off lead geometry (the constant deceleration needed to match the lead's speed by a
small gap), not the planner's acceleration request, which is shaped for cars that can brake and would
fire far too early on this one.

  attention  yellow banner + one chime: a slower car ahead needs about what lowering the set speed gives
  backstop   cancel the stock cruise, feed the regen paddle while needed, press RES once the gap is rebuilt
  brake      red BRAKE! + loud tone: more than the paddle can give, or the paddle is not gaining; driver must brake

openpilot stays engaged through the backstop; the car controller reads the published action and sends
the frames, and the panda only permits them for a short window after openpilot's own cancel.
"""
from dataclasses import dataclass

from opendbc.car.gm.values import CAR as GM_CAR

CRUISE_BUTTON_BRAKE_CARS = {GM_CAR.CHEVROLET_BOLT_CC_2017, GM_CAR.CHEVROLET_BOLT_CC_2018_2021, GM_CAR.CHEVROLET_BOLT_CC_2022_2023}

SET_SPEED_DECEL_AUTHORITY = 0.56     # m/s^2, measured
PADDLE_DECEL_AUTHORITY = 1.3         # m/s^2, measured

STAGE_IDLE = 0
STAGE_ATTENTION = 1
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

MIN_SPEED = 5.0                      # m/s, nothing below this
LEAD_MIN_PROB = 0.85                 # vision leads only count when the model is confident
LEAD_MIN_CLOSING = 3.0               # m/s
LEAD_GAP = 8.0                       # m, gap the required-decel estimate aims to keep

# tier 1: heads-up
ATTENTION_REQUIRED_DECEL = 0.45      # about what the set speed can give
ATTENTION_RELEASE_DECEL = 0.25
ATTENTION_RELEASE_S = 1.5

# backstop
CANCEL_MIN_SPEED = 8.0               # m/s
CANCEL_REQUIRED_DECEL = 0.75         # clearly beyond the set speed
CANCEL_CONFIRM_S = 0.5
CANCEL_IMMEDIATE_REQUIRED_DECEL = 1.2
CANCEL_ACCEL_BACKUP = -1.5           # no usable lead geometry but a strong, sustained planner request
CANCEL_ACCEL_BACKUP_S = 1.5
CANCEL_SETTLE_S = 0.5                # after cruise drops: still needed -> paddle, eased -> resume
CANCEL_TIMEOUT_S = 1.5               # cruise did not drop: the paddle cancels it as well
NEEDED_REQUIRED_DECEL = 0.6          # keep / return to the paddle
NEEDED_ACCEL = -1.2
EASED_REQUIRED_DECEL = 0.4
EASED_ACCEL = -0.5

PADDLE_MIN_SPEED = 3.0               # m/s
PADDLE_MIN_HOLD_S = 1.0
PADDLE_MAX_HOLD_S = 8.0
PADDLE_RELEASE_S = 0.5

RESUME_SETTLE_S = 1.0                # eased this long before pressing RES
RESUME_TIMEOUT_S = 3.0               # RES presses not taking
RESUME_GIVEUP_S = 12.0               # conditions never right (too slow, gas held): hand back to the driver
RESUME_LOCKOUT_S = 3.0               # no new cancel right after a resume
BACKSTOP_WINDOW_S = 18.0             # stay inside the panda's 20 s window after our cancel

# tier 2: driver must brake
BRAKE_REQUIRED_DECEL = 1.5           # beyond the paddle
BRAKE_MIN_TTC_S = 2.5
BRAKE_PADDLE_STALL_S = 2.0           # paddle held this long and the lead still needs more than the set speed can give
BRAKE_PADDLE_STALL_REQUIRED_DECEL = 1.1
BRAKE_RELEASE_REQUIRED_DECEL = 1.0
BRAKE_RELEASE_S = 1.0

DISENGAGE_EVENT_S = 0.3              # hold the hand-back event so selfdrived cannot miss it
TIMER_EPS = 1e-6


@dataclass
class BrakeOutputs:
  attention: bool = False
  attention_chime: bool = False
  brake: bool = False
  stage: int = STAGE_IDLE
  action: int = ACTION_NONE
  disengage: bool = False


def required_decel_to_match_lead(v_ego, lead_d_rel, lead_v_lead, gap=LEAD_GAP):
  """Constant deceleration needed to be at the lead's speed once the gap has shrunk to `gap`."""
  closing = float(v_ego) - float(lead_v_lead)
  if closing < LEAD_MIN_CLOSING:
    return 0.0
  room = max(float(lead_d_rel) - gap, 1.0)
  return closing * closing / (2.0 * room)


def time_to_contact(v_ego, lead_d_rel, lead_v_lead):
  closing = float(v_ego) - float(lead_v_lead)
  if closing < LEAD_MIN_CLOSING:
    return float("inf")
  return max(float(lead_d_rel), 0.0) / closing


class CruiseButtonBrake:
  """Alert tiers and the cancel -> paddle -> resume stage machine. Call update() every frame."""

  def __init__(self):
    self.reset()

  def reset(self):
    self.stage = STAGE_IDLE
    self.attention = False
    self.attention_release_s = 0.0
    self.brake = False
    self.brake_release_s = 0.0
    self.cancel_s = 0.0
    self.accel_backup_s = 0.0
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
    return BrakeOutputs(disengage=True)

  def update(self, dt, *, enabled, cruise_active, cruise_available, v_ego, accel_cmd, lead_status, lead_d_rel,
             lead_v_lead, lead_prob, gas_pressed, brake_pressed, driver_regen, backstop_allowed, min_resume_speed):
    if not enabled:
      self.reset()
      return BrakeOutputs()

    if self.disengage_s > TIMER_EPS:
      self.disengage_s -= dt
      return BrakeOutputs(disengage=True)

    lead_ok = lead_status and float(lead_prob) >= LEAD_MIN_PROB
    required = required_decel_to_match_lead(v_ego, lead_d_rel, lead_v_lead) if lead_ok else 0.0
    ttc = time_to_contact(v_ego, lead_d_rel, lead_v_lead) if lead_ok else float("inf")
    needed = required >= NEEDED_REQUIRED_DECEL or accel_cmd <= NEEDED_ACCEL
    eased = required < EASED_REQUIRED_DECEL and accel_cmd > EASED_ACCEL
    driver_override = gas_pressed or brake_pressed or driver_regen

    if self.stage in (STAGE_IDLE, STAGE_ATTENTION) and v_ego < MIN_SPEED:
      self.reset()
      return BrakeOutputs()

    self.lockout_s = max(self.lockout_s - dt, 0.0)
    if self.stage >= STAGE_CANCEL:
      self.since_cancel_s += dt
    self.stage_s += dt

    # --- tier 1: attention (hysteresis, one chime on the rising edge)
    chime = False
    if not self.attention:
      if required >= ATTENTION_REQUIRED_DECEL:
        self.attention = True
        self.attention_release_s = 0.0
        chime = True
    else:
      self.attention_release_s = self.attention_release_s + dt if required < ATTENTION_RELEASE_DECEL else 0.0
      if self.attention_release_s + TIMER_EPS >= ATTENTION_RELEASE_S:
        self.attention = False
    if self.stage == STAGE_IDLE and self.attention:
      self.stage = STAGE_ATTENTION
    elif self.stage == STAGE_ATTENTION and not self.attention:
      self.stage = STAGE_IDLE

    # --- tier 2: brake (driver action required), independent of the stage machine
    paddle_stalled = (self.stage == STAGE_PADDLE and self.hold_s + TIMER_EPS >= BRAKE_PADDLE_STALL_S and
                      required >= BRAKE_PADDLE_STALL_REQUIRED_DECEL)
    brake_trigger = required >= BRAKE_REQUIRED_DECEL or ttc <= BRAKE_MIN_TTC_S or paddle_stalled
    if not self.brake:
      if brake_trigger:
        self.brake = True
        self.brake_release_s = 0.0
    else:
      self.brake_release_s = self.brake_release_s + dt if (required < BRAKE_RELEASE_REQUIRED_DECEL and not brake_trigger) else 0.0
      if self.brake_release_s + TIMER_EPS >= BRAKE_RELEASE_S:
        self.brake = False

    # --- backstop stage machine
    action = ACTION_NONE
    self.cancel_s = self.cancel_s + dt if required >= CANCEL_REQUIRED_DECEL else 0.0
    self.accel_backup_s = self.accel_backup_s + dt if accel_cmd <= CANCEL_ACCEL_BACKUP else 0.0

    if self.stage in (STAGE_IDLE, STAGE_ATTENTION):
      can_cancel = (backstop_allowed and cruise_active and not driver_override and
                    v_ego >= CANCEL_MIN_SPEED and self.lockout_s <= TIMER_EPS)
      trigger = (self.cancel_s + TIMER_EPS >= CANCEL_CONFIRM_S or
                 required >= CANCEL_IMMEDIATE_REQUIRED_DECEL or
                 self.accel_backup_s + TIMER_EPS >= CANCEL_ACCEL_BACKUP_S)
      if can_cancel and trigger:
        self.attention = True
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
        self.lockout_s = RESUME_LOCKOUT_S
        self.stage = STAGE_ATTENTION if self.attention else STAGE_IDLE
        self.stage_s = 0.0
        return BrakeOutputs(attention=self.attention, brake=self.brake, stage=self.stage, action=ACTION_NONE)
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

    return BrakeOutputs(attention=self.attention, attention_chime=chime, brake=self.brake, stage=self.stage, action=action)
