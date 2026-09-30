"""Action channel from starpilot's cruise-button brake stage machine to the GM car controller.

A plain file in shared memory rather than a Params key: the key table is compiled into the prebuilt
params module, so a new key does not exist on a device until that module is rebuilt. The value carries
a monotonic timestamp and the reader treats anything older than STALE_S as "no action", so a stalled
writer can never leave the paddle or a button held.
"""
import os
import tempfile
import time

from openpilot.system.hardware.hw import Paths

FILENAME = "starpilot_cruise_button_brake_action"
STALE_S = 0.5


def channel_path() -> str:
  base = Paths.shm_path()
  if not os.path.isdir(base):
    base = tempfile.gettempdir()
  return os.path.join(base, FILENAME)


def write_action(action: int, now: float | None = None, path: str | None = None) -> None:
  path = path or channel_path()
  now = time.monotonic() if now is None else now
  tmp = f"{path}.{os.getpid()}.tmp"
  with open(tmp, "w", encoding="utf-8") as f:
    f.write(f"{int(action)} {now:.3f}")
  os.replace(tmp, path)


def read_action(now: float | None = None, path: str | None = None) -> int:
  path = path or channel_path()
  now = time.monotonic() if now is None else now
  try:
    with open(path, encoding="utf-8") as f:
      raw = f.read().split()
    action, stamp = int(raw[0]), float(raw[1])
  except (OSError, ValueError, IndexError):
    return 0
  if now - stamp > STALE_S or now < stamp - STALE_S:
    return 0
  return action
