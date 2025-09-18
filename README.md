# Mobile Landing Pad for MAVs (ROS 2)

Autonomous landing on a moving, ArUco‑tagged platform using **ROS 2**, **Intel RealSense T265**, **OpenCV ArUco**, and **PX4/MAVROS**. The node estimates the pad pose from fisheye images, transforms it into the world frame, and commands position setpoints for a smooth approach and landing.

> This repo includes the ROS 2 Python node (`land_at_aruco.py`) and the final design report PDF in `report/`.

## Features
- ArUco detection with `opencv-contrib` (PnP) and fisheye camera calibration
- Real‑time transform from RealSense to Vicon/world frame
- Simple state machine: **INIT → LAUNCH → TEST (approach) → LAND → ABORT**
- Publishes setpoints to PX4 via **MAVROS**

## Repository layout
```
.
├─ land_at_aruco.py
├─ report/
│  └─ ROB498_Final_Report_Mobile_Platform_for_MAVs.pdf
├─ requirements.txt
├─ .gitignore
└─ LICENSE
```

## Prerequisites
- Ubuntu + **ROS 2** (Humble/Foxy) with `rclpy`
- **MAVROS** installed and connected to PX4 (Offboard enabled)
- Intel **RealSense T265** with `realsense2_camera` publishing odometry
- Python 3.8+ with packages in `requirements.txt`

> `rclpy`, message types, and MAVROS come from your ROS 2 install; Python packages below are for the vision pieces.

```bash
python3 -m pip install -r requirements.txt
```

## Topics & Services

**Subscribes**
- `/vicon/ROB498_Drone/ROB498_Drone` (`geometry_msgs/PoseStamped`) – global pose
- `/camera/pose/sample` (`nav_msgs/Odometry`) – RealSense T265 odom
- `/camera/fisheye1/image_raw` (`sensor_msgs/Image`) – fisheye grayscale

**Publishes**
- `/mavros/setpoint_position/local` (`geometry_msgs/PoseStamped`)
- `/mavros/vision_pose/pose` (`geometry_msgs/PoseStamped`)
- `aruco_vector` (`geometry_msgs/PoseStamped`) – relative vector to pad

**Services (Trigger)**
- `/rob498_drone_13/comm/launch`
- `/rob498_drone_13/comm/test`
- `/rob498_drone_13/comm/land`
- `/rob498_drone_13/comm/abort`

## Quickstart
Source ROS 2 and your workspace (MAVROS running), then run:

```bash
# Terminal A – launch your PX4/MAVROS + RealSense drivers
# Terminal B – run the node
python3 land_at_aruco.py
```

Call services from another terminal as you test:
```bash
ros2 service call /rob498_drone_13/comm/launch std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_13/comm/test   std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_13/comm/land   std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_13/comm/abort  std_srvs/srv/Trigger {}
```

## Calibration
Camera intrinsics (`cameraMatrix`, `distCoeffs`) and **marker_length** live at the top of `land_at_aruco.py`. Update them to your camera/marker.

## Notes
- The RealSense T265 publishes grayscale fisheye frames; ArUco detection here uses the new OpenCV ArUco API (`ArucoDetector`).  
- If you prefer a full ROS 2 package later, add `package.xml`, `setup.py`, and a `launch/` file, then export console entry points to run with `ros2 run`.

## License
MIT — see `LICENSE`.
