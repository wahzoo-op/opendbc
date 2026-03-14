import numpy as np
from opendbc.car import Bus, get_safety_config, structs
from opendbc.car.carlog import carlog
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.ford.carcontroller import CarController, APA_ANGLE_MRAD_MAX
from opendbc.car.ford.carstate import CarState
from opendbc.car.ford.fordcan import CanBus
from opendbc.car.ford.radar_interface import RadarInterface
from opendbc.car.ford.values import CarControllerParams, DBC, Ecu, FordFlags, RADAR, FordSafetyFlags
from opendbc.car.interfaces import CarInterfaceBase

TransmissionType = structs.CarParams.TransmissionType

# APA max steering angle in degrees (102.35 mrad converted)
APA_MAX_STEER_DEG = APA_ANGLE_MRAD_MAX / 1000.0 * 180.0 / np.pi  # ~5.86 degrees


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController
  RadarInterface = RadarInterface

  @staticmethod
  def get_steer_feedforward_apa(desired_angle, v_ego):
    # Normalize desired angle to the APA signal range (±5.86°).
    # This provides the baseline command to hold the desired angle.
    # The PID P/I terms add corrections for tracking error.
    return desired_angle / APA_MAX_STEER_DEG

  def get_steer_feedforward_function(self):
    if self.CP.flags & FordFlags.APA:
      return self.get_steer_feedforward_apa
    return CarInterfaceBase.get_steer_feedforward_default

  @staticmethod
  def get_pid_accel_limits(CP, current_speed, cruise_speed):
    # PCM doesn't allow acceleration near cruise_speed,
    # so limit limits of pid to prevent windup
    ACCEL_MAX_VALS = [CarControllerParams.ACCEL_MAX, 0.2]
    ACCEL_MAX_BP = [cruise_speed - 2., cruise_speed - .4]
    return CarControllerParams.ACCEL_MIN, np.interp(current_speed, ACCEL_MAX_BP, ACCEL_MAX_VALS)

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = "ford"

    ret.radarUnavailable = Bus.radar not in DBC[candidate]
    ret.steerControlType = structs.CarParams.SteerControlType.angle
    ret.steerActuatorDelay = 0.2
    ret.steerLimitTimer = 1.0
    ret.steerAtStandstill = True

    ret.longitudinalTuning.kiBP = [0.]
    ret.longitudinalTuning.kiV = [0.5]

    if not ret.radarUnavailable and DBC[candidate][Bus.radar] == RADAR.DELPHI_MRR:
      # average of 33.3 Hz radar timestep / 4 scan modes = 60 ms
      # MRR_Header_Timestamps->CAN_DET_TIME_SINCE_MEAS reports 61.3 ms
      ret.radarDelay = 0.06

    CAN = CanBus(fingerprint=fingerprint)
    cfgs = [get_safety_config(structs.CarParams.SafetyModel.ford)]
    if CAN.main >= 4:
      cfgs.insert(0, get_safety_config(structs.CarParams.SafetyModel.noOutput))
    ret.safetyConfigs = cfgs

    ret.alphaLongitudinalAvailable = ret.radarUnavailable
    if alpha_long or not ret.radarUnavailable:
      ret.safetyConfigs[-1].safetyParam |= FordSafetyFlags.LONG_CONTROL.value
      ret.openpilotLongitudinalControl = True

    if ret.flags & FordFlags.APA:
      ret.steerActuatorDelay = 0.6
      ret.safetyConfigs[-1].safetyParam |= FordSafetyFlags.APA.value

      # APA uses PID lateral control instead of pure angle feedforward.
      # LatControlAngle has no feedback loop — it sends the desired angle directly.
      # When desired angles exceed the ±5.86° APA signal range, the command saturates,
      # the PSCM overshoots, and the model overcorrects → growing oscillation.
      # PID with feedforward provides damping: feedforward holds the angle,
      # P corrects tracking errors, I eliminates steady-state offset.
      # Gains inspired by the old C2 branch's INDI tuning (outerLoopGain=3.5, timeConstant=2.0).
      ret.steerControlType = structs.CarParams.SteerControlType.torque
      ret.lateralTuning.pid.kpBP = [0.]
      ret.lateralTuning.pid.kpV = [0.005]
      ret.lateralTuning.pid.kiBP = [0.]
      ret.lateralTuning.pid.kiV = [0.001]
      ret.lateralTuning.pid.kf = 0.15  # feedforward scales desired_angle / APA_MAX_STEER_DEG


    if ret.flags & FordFlags.CANFD:
      ret.safetyConfigs[-1].safetyParam |= FordSafetyFlags.CANFD.value

      # TRON (SecOC) platforms are not supported
      # LateralMotionControl2, ACCDATA are 16 bytes on these platforms
      if len(fingerprint[CAN.camera]):
        if fingerprint[CAN.camera].get(0x3d6) != 8 or fingerprint[CAN.camera].get(0x186) != 8:
          carlog.error('dashcamOnly: SecOC is unsupported')
          ret.dashcamOnly = True
    else:
      # Lock out if the car does not have needed lateral and longitudinal control APIs.
      # Note that we also check CAN for adaptive cruise, but no known signal for LCA exists
      # APA and PINION_ALT cars skip the LCA/TJA firmware check (Edge may not have TJA configured).
      if not (ret.flags & (FordFlags.APA | FordFlags.PINION_ALT)):
        pscm_config = next((fw for fw in car_fw if fw.ecu == Ecu.eps and b'\x22\xDE\x01' in fw.request), None)
        if pscm_config:
          if len(pscm_config.fwVersion) != 24:
            carlog.error('dashcamOnly: Invalid EPS FW version')
            ret.dashcamOnly = True
          else:
            config_tja = pscm_config.fwVersion[7]  # Traffic Jam Assist
            config_lca = pscm_config.fwVersion[8]  # Lane Centering Assist
            if config_tja != 0xFF or config_lca != 0xFF:
              carlog.error('dashcamOnly: Car lacks required lateral control APIs')
              ret.dashcamOnly = True

    # Auto Transmission: 0x732 ECU or Gear_Shift_by_Wire_FD1
    found_ecus = [fw.ecu for fw in car_fw]
    if Ecu.shiftByWire in found_ecus or 0x5A in fingerprint[CAN.main] or docs:
      ret.transmissionType = TransmissionType.automatic
    else:
      ret.transmissionType = TransmissionType.manual
      ret.minEnableSpeed = 20.0 * CV.MPH_TO_MS

    # BSM: Side_Detect_L_Stat, Side_Detect_R_Stat
    # TODO: detect bsm in car_fw?
    ret.enableBsm = 0x3A6 in fingerprint[CAN.main] and 0x3A7 in fingerprint[CAN.main]

    # LCA can steer down to zero
    ret.minSteerSpeed = 0.

    ret.autoResumeSng = ret.minEnableSpeed == -1.
    ret.centerToFront = ret.wheelbase * 0.44
    return ret
