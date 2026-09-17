"""Brake-now alert and L-mode cancel backstop for cruise-button Bolts (no ACC, no pedal interceptor).

Measured on a 2023 Bolt EUV, 65 -> 55 mph:
  lowering the set speed   ~8 s   (0.56 m/s^2, same in D and L: the stock cruise uses its own mild regen)
  cancelling cruise in L   ~4 s   (1.1 m/s^2, driver lift-off regen in L)
  regen paddle             ~3.5 s (1.3 m/s^2)
Lowering the set speed is the weakest brake the car has. When the planner needs clearly more than that,
warn the driver; when it keeps needing more and the car is in L, cancel cruise so lift-off regen takes
over. openpilot disengages with the cancel (always-on lateral keeps steering) and the driver resumes
with RES.
"""
from opendbc.car.gm.values import CAR as GM_CAR

CRUISE_BUTTON_BRAKE_CARS = {GM_CAR.CHEVROLET_BOLT_CC_2017, GM_CAR.CHEVROLET_BOLT_CC_2018_2021, GM_CAR.CHEVROLET_BOLT_CC_2022_2023}

SET_SPEED_DECEL_AUTHORITY = 0.56     # m/s^2, measured
L_MODE_CANCEL_DECEL_AUTHORITY = 1.1  # m/s^2, measured

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
ALERT_HOLD_AFTER_CANCEL_S = 6.0      # keep warning after the backstop cancelled cruise while the lead still needs braking

CANCEL_MIN_SPEED = 8.0               # m/s
CANCEL_ALERT_LEAD_S = 0.7            # the alert must have been up this long first
CANCEL_ACCEL = -1.0
CANCEL_REQUIRED_DECEL = 0.9
CANCEL_IMMEDIATE_REQUIRED_DECEL = 1.5
CANCEL_EVENT_S = 0.3                 # hold the disable event so selfdrived cannot miss it
TIMER_EPS = 1e-6                     # timers accumulate dt; compare with a little slack


def required_decel_to_match_lead(v_ego, lead_d_rel, lead_v_lead, gap=ALERT_GAP):
  """Constant deceleration needed to be at the lead's speed once the gap has shrunk to `gap`."""
  closing = float(v_ego) - float(lead_v_lead)
  if closing < ALERT_MIN_CLOSING:
    return 0.0
  room = max(float(lead_d_rel) - gap, 1.0)
  return closing * closing / (2.0 * room)


class CruiseButtonBrake:
  """Hysteresis state for the alert and the cancel backstop. Call update() every frame with its dt."""

  def __init__(self):
    self.reset()

  def reset(self):
    self.alert = False
    self.accel_s = 0.0
    self.release_s = 0.0
    self.alert_s = 0.0
    self.cancelled = False
    self.cancel_event_s = 0.0
    self.hold_s = 0.0

  def update(self, dt, *, enabled, cruise_active, v_ego, accel_cmd, lead_status, lead_d_rel, lead_v_lead,
             lead_prob, gear_low, driver_input, cancel_allowed):
    """Returns (show_alert, request_cancel)."""
    required = 0.0
    if lead_status and float(lead_prob) >= ALERT_MIN_LEAD_PROB:
      required = required_decel_to_match_lead(v_ego, lead_d_rel, lead_v_lead)

    if not enabled:
      # After our own cancel, keep the warning up while the lead still needs real braking.
      if self.cancelled and self.hold_s > TIMER_EPS and required >= ALERT_RELEASE_DECEL:
        self.hold_s -= dt
        self.cancel_event_s = max(self.cancel_event_s - dt, 0.0)
        return True, self.cancel_event_s > TIMER_EPS
      self.reset()
      return False, False

    if v_ego < ALERT_MIN_SPEED:
      self.reset()
      return False, False

    self.accel_s = self.accel_s + dt if accel_cmd <= ALERT_ACCEL else 0.0
    accel_trigger = self.accel_s + TIMER_EPS >= ALERT_ACCEL_CONFIRM_S
    lead_trigger = required >= ALERT_REQUIRED_DECEL

    if not self.alert:
      if accel_trigger or lead_trigger:
        self.alert = True
        self.alert_s = 0.0
        self.release_s = 0.0
    else:
      self.alert_s += dt
      eased = accel_cmd > ALERT_RELEASE_ACCEL and required < ALERT_RELEASE_DECEL
      self.release_s = self.release_s + dt if eased else 0.0
      if self.release_s + TIMER_EPS >= ALERT_RELEASE_S:
        self.alert = False
        self.accel_s = 0.0
        self.alert_s = 0.0

    cancel = False
    if self.cancelled:
      self.cancel_event_s = max(self.cancel_event_s - dt, 0.0)
      cancel = self.cancel_event_s > TIMER_EPS
    elif cancel_allowed and gear_low and cruise_active and not driver_input and v_ego >= CANCEL_MIN_SPEED:
      sustained = (self.alert and self.alert_s + TIMER_EPS >= CANCEL_ALERT_LEAD_S and
                   (accel_cmd <= CANCEL_ACCEL or required >= CANCEL_REQUIRED_DECEL))
      if sustained or required >= CANCEL_IMMEDIATE_REQUIRED_DECEL:
        self.cancelled = True
        self.alert = True
        self.cancel_event_s = CANCEL_EVENT_S
        self.hold_s = ALERT_HOLD_AFTER_CANCEL_S
        cancel = True

    return self.alert, cancel
