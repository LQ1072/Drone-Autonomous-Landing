# Drone-Autonomous-Landing
Autonomous landing of a PX4-based MAV on a **moving** platform using an **Intel RealSense T265**, **OpenCV ArUco**, and **MAVROS**.   The node detects the landing marker, estimates its 3D pose, transforms it into the world frame, and publishes setpoints for approach/landing.
