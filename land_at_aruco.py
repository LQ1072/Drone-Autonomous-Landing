import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, PoseArray
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import tf_transformations as tfs
from tf_transformations import quaternion_matrix, quaternion_from_matrix, inverse_matrix

import copy

# rs aruco imports
import cv2
import cv2.aruco as aruco
import time

# aruco constants

# CAMERA MATRIX
cameraMatrix = np.array([[287.793, 0, 411.906],
                         [0, 287.798, 399.433],
                         [0, 0, 1]])
distCoeffs = np.array([[-0.0097255], [0.0474876], [-0.0441816], [0.0080753], [0]])

# Define physical marker size (in meters)
marker_length = 0.2  # 100 mm marker

# Define the marker's 3D object points (with the center at [0,0,0])
object_points = np.array([
    [-marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32)

# Flight state constants
INIT = 0
LAUNCH = 1
TEST = 2
LAND = 3
ABORT = 4

# Z-offset between Vicon marker origin and drone's base (e.g., marker is 0.13 m above ground)
Z_OFFSET = -0.13


class CommNode(Node):
    def __init__(self):
        super().__init__('rob498_drone_13')  # Use team ID in node name
        # Service servers for flight commands
        self.srv_launch = self.create_service(Trigger, 'rob498_drone_13/comm/launch', self.callback_launch)
        self.srv_test   = self.create_service(Trigger, 'rob498_drone_13/comm/test', self.callback_test)
        self.srv_land   = self.create_service(Trigger, 'rob498_drone_13/comm/land', self.callback_land)
        self.srv_abort  = self.create_service(Trigger, 'rob498_drone_13/comm/abort', self.callback_abort)

        # Internal flags
        self.current_state = INIT
        self.is_switched = False        # Indicates a state switch (used for TEST start)

        # Subscriptions to sensor inputs
        self.create_subscription(PoseStamped, '/vicon/ROB498_Drone/ROB498_Drone', self.vicon_callback, 10)

        self.create_subscription(Odometry, '/camera/pose/sample', self.realsense_callback, rclpy.qos.qos_profile_system_default)

        self.create_subscription(Image, '/camera/fisheye1/image_raw', self.aruco_callback, rclpy.qos.qos_profile_system_default)


        # Publishers for setpoint and current pose (to FCU via MAVROS)
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.curr_pose_pub = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 10)

        self.drone_to_pad_rsf_pub = self.create_publisher(PoseStamped, 'aruco_vector', 10)

        # Storage for sensor data and waypoints
        self.last_vicon = None
        self.last_realsense = None
        self.origin_setpoint = None     # PoseStamped of initial pose (Vicon frame) on ground
        self.final_setpoint = None      # Current target setpoint PoseStamped (Vicon frame)

        # Storage for aruco data
        #self.curr_landpad_rsf = None
        self.vec_rs_to_landpad_rsf = None

        # Transformation matrix from Realsense frame to Vicon frame (4x4 homogeneous)
        self.rs2vicon_H = np.eye(4)

        # Timer at 20 Hz for publishing setpoints and handling state machine
        self.timer = self.create_timer(0.05, self.timer_callback)

    # Service callbacks for flight commands
    def callback_launch(self, request, response):
        """Handle 'launch' command: ascend to test altitude and enter LAUNCH state."""
        if self.origin_setpoint is not None:
            # Set target takeoff position (keep x, y at 0,0 in Vicon frame, z to 1.5 m above ground)
            setpoint = PoseStamped()
            setpoint.header.frame_id = 'map'
            setpoint.header.stamp = self.get_clock().now().to_msg()
            # Use initial orientation (yaw) from origin_setpoint
            setpoint.pose.orientation = self.origin_setpoint.pose.orientation
            setpoint.pose.position.x = self.origin_setpoint.pose.position.x
            setpoint.pose.position.y = self.origin_setpoint.pose.position.y
            setpoint.pose.position.z = self.origin_setpoint.pose.position.z + 0.8
            self.final_setpoint = setpoint

        self.current_state = LAUNCH
        response.success = True
        response.message = "Launch command received"
        print("LAUNCH command received – entering LAUNCH state")
        return response

    def callback_test(self, request, response):
        """Handle 'test' command: begin waypoint navigation sequence (TEST state)."""
        
        self.current_state = TEST
        response.success = True
        response.message = "Test command received"
        print("TEST command received – entering TEST state")
        return response

    def callback_land(self, request, response):
        """Handle 'land' command: initiate landing sequence (LAND state)."""
        if self.origin_setpoint is not None:
            # Target the original takeoff position (ground level) for landing
            setpoint = PoseStamped()
            setpoint.header.frame_id = 'map'
            setpoint.header.stamp = self.get_clock().now().to_msg()
            setpoint.pose.orientation = self.last_vicon.pose.orientation
            setpoint.pose.position = Point(
                x=self.last_vicon.pose.position.x,
                y=self.last_vicon.pose.position.y,
                z=self.last_vicon.pose.position.z - 0.4
            )
            self.final_setpoint = setpoint
        
        self.current_state = LAND
        response.success = True
        response.message = "Land command received"
        print("LAND command received – entering LAND state")
        return response

    def callback_abort(self, request, response):
        """Handle 'abort' command: emergency abort (ABORT state)."""
        self.current_state = ABORT
        # Immediately send disarm command to shut down motors
        response.success = True
        response.message = "Abort command received"
        print("ABORT command received – entering ABORT state (disarming)")
        return response

    # Vicon pose callback
    def vicon_callback(self, msg: PoseStamped):
        """Receive Vicon PoseStamped and update last_vicon reading (in 'map' frame)."""
        # Update last_vicon pose and timestamp
        self.last_vicon = msg
        self.last_vicon.header.stamp = self.get_clock().now().to_msg()
        self.last_vicon.header.frame_id = 'map'

    # Realsense odometry callback
    def realsense_callback(self, msg: Odometry):
        """Receive Realsense Odometry and update last_realsense as PoseStamped (in 'map' frame)."""
        # Convert Odometry to PoseStamped
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = Pose(
            position=Point(x=msg.pose.pose.position.x,
                           y=msg.pose.pose.position.y,
                           z=msg.pose.pose.position.z),
            orientation=Quaternion(x=msg.pose.pose.orientation.x,
                                   y=msg.pose.pose.orientation.y,
                                   z=msg.pose.pose.orientation.z,
                                   w=msg.pose.pose.orientation.w)
        )
        self.last_realsense = ps

    def aruco_callback(self, msg: Image):

        if self.last_realsense is None:
            return

        # output is in rs frame
        img = np.reshape(msg.data, (msg.height, msg.width))
        vec_rs_to_aruco_rsf = self.aruco_conversion(img)
        
        if vec_rs_to_aruco_rsf is not None:

            self.vec_rs_to_landpad_rsf = copy.deepcopy(self.last_realsense)

            # hardcoded transform to realsense frame from aruco frame
            self.vec_rs_to_landpad_rsf.pose.position.x = (vec_rs_to_aruco_rsf[2] - 0.5)
            self.vec_rs_to_landpad_rsf.pose.position.y = -vec_rs_to_aruco_rsf[0]
            self.vec_rs_to_landpad_rsf.pose.position.z = -vec_rs_to_aruco_rsf[1]

            #self.vec_rs_to_landpad_rsf.pose.position.x =  vec_rs_to_aruco_rsf[2]
            #self.vec_rs_to_landpad_rsf.pose.position.y = -vec_rs_to_aruco_rsf[0]
            #self.vec_rs_to_landpad_rsf.pose.position.z = -vec_rs_to_aruco_rsf[1]

            self.drone_to_pad_rsf_pub.publish(self.vec_rs_to_landpad_rsf)

            print("ARUCO SEEN, WAITING FOR TEST CMD")
            #print(self.vec_rs_to_landpad_rsf.pose.position)
            #print(self.last_realsense.pose.position)
            #print()

    def rs2vicon_frame(self):
        """Compute the homogeneous transform from Realsense frame to Vicon frame using last known poses."""
        rs_H = self.quat2tfmatrix(self.last_realsense)
        vicon_H = self.quat2tfmatrix(self.last_vicon)
        #print("checking 4x4?")
        #print(vicon_H)
        inv_rs_H = inverse_matrix(rs_H)
        # rs2vicon_H transforms a pose in Realsense coordinates to Vicon coordinates
        self.rs2vicon_H = np.matmul(vicon_H, inv_rs_H)

    def quat2tfmatrix(self, pose_msg: PoseStamped):
        """Convert a PoseStamped into a 4x4 transformation matrix."""
        q = pose_msg.pose.orientation
        p = pose_msg.pose.position
        # Use tf_transformations to get rotation matrix from quaternion
        T = quaternion_matrix([q.x, q.y, q.z, q.w])
        # Insert translation
        T[0:3, 3] = np.array([p.x, p.y, p.z])
        return T

    def tfmatrix2pose(self, H):
        """Convert a 4x4 transformation matrix into a PoseStamped (in 'map' frame)."""
        q = quaternion_from_matrix(H)
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'map'
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position = Point(x=H[0, 3], y=H[1, 3], z=H[2, 3])
        pose_msg.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return pose_msg

    def aruco_conversion(self, rect_image):

        # Get the predefined dictionary for 6x6 markers (250 markers available)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()  # Using default parameters
        # Create an ArucoDetector object using the new API
        detector = aruco.ArucoDetector(aruco_dict, parameters)
    
        # Timer for updating the display text (once per second)
        last_update_time = time.time()
    
        # Variable to store the last computed marker 3D position (in camera frame)
        display_marker_position = None
    
        frame = rect_image

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = frame

        corners, ids, _ = detector.detectMarkers(gray)

        marker_position_3d = None

        if ids is not None:
            # Draw the detected markers (for visualization)
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            # Process each detected marker
            for marker_corners in corners:
                image_points = marker_corners.reshape((4, 2)).astype(np.float32)
                ret, rvec, tvec = cv2.solvePnP(object_points, image_points, cameraMatrix, distCoeffs)
                if ret:
                    # Draw coordinate axes for visualization (axes length set to 5 cm)
                    cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.05, 2)
    
                    # tvec represents the marker's center in the camera coordinate system.
                    # Since object_points were defined relative to the marker center (0,0,0),
                    # tvec (after flattening) is [X, Y, Z] where:
                    #   X is right (in meters), Y is down, and Z is forward.
                    marker_position_3d = tvec.flatten()
    
                    current_time = time.time()
                    # Update the displayed 3D position every second
                    display_marker_position = marker_position_3d
                    #print("t = {:.2f} : Marker 3D Position (Camera Frame): X={:.2f} m, Y={:.2f} m, Z={:.2f} m".format(
                    #    current_time,
                    #    marker_position_3d[0],
                    #    marker_position_3d[1],
                    #    marker_position_3d[2]
                    #))
    
        return marker_position_3d
    

    def timer_callback(self):
        """Main loop executed at 20 Hz to publish setpoints and manage flight state transitions."""

        # TESTING ARUCO PURPOSES ONLY
        #if self.last_realsense is None:
        #    return
        #if self.vec_rs_to_landpad_rsf is None:
        #    return
        #print("x = {:.5f} y = {:.5f} z = {:.5f}".format(
        #        self.vec_rs_to_landpad_rsf.pose.position.x, 
        #        self.vec_rs_to_landpad_rsf.pose.position.y, 
        #        self.vec_rs_to_landpad_rsf.pose.position.z))
        #print()
        #return


        # If in ABORT state, do nothing (motors disarmed)
        if self.current_state == ABORT:
            return


        # Ensure we have initial pose data from both Vicon and Realsense before controlling
        if self.last_vicon is None or self.last_realsense is None:
            print("Waiting for Vicon and Realsense data...")
            return

        if self.current_state == INIT:
            # Continuously update transform calibration while waiting in INIT
            self.rs2vicon_frame()
            # Set origin_setpoint to the latest Vicon pose (initial position)
            if self.origin_setpoint is None:
                self.origin_setpoint = PoseStamped()
                self.origin_setpoint.header = Header(frame_id='map', stamp=self.get_clock().now().to_msg())
                self.origin_setpoint.pose = Pose(
                    position=Point(x=self.last_vicon.pose.position.x,
                                   y=self.last_vicon.pose.position.y,
                                   z=self.last_vicon.pose.position.z),
                    orientation=self.last_vicon.pose.orientation
                )
            # During INIT, publish current pose as both setpoint and vision pose (hold position)
            self.setpoint_pub.publish(self.origin_setpoint)
            self.curr_pose_pub.publish(self.last_vicon)


        elif self.current_state in (LAUNCH, LAND):

            # Publish current target setpoint and current pose to MAVROS
            if self.final_setpoint is not None:
                # Update timestamp on setpoint before publishing
                self.final_setpoint.header.stamp = self.get_clock().now().to_msg()
                self.setpoint_pub.publish(self.final_setpoint)
            # Publish the current pose (feedback) in vision_pose topic
            self.curr_pose_pub.publish(self.last_vicon)


        elif self.current_state == TEST:
        
            curr_landpad_rsf_H = self.quat2tfmatrix(self.vec_rs_to_landpad_rsf)
            curr_landpad_viconf_H = np.matmul(self.rs2vicon_H, curr_landpad_rsf_H)
            curr_landpad_viconf_pose = self.tfmatrix2pose(curr_landpad_viconf_H)

            # Target the original takeoff position (ground level) for landing
            setpoint = PoseStamped()
            setpoint.header.frame_id = 'map'
            setpoint.header.stamp = self.get_clock().now().to_msg()
            setpoint.pose.orientation = self.origin_setpoint.pose.orientation
            setpoint.pose.position = Point(
                x=self.last_vicon.pose.position.x + curr_landpad_viconf_pose.pose.position.x, 
                y=self.last_vicon.pose.position.y + curr_landpad_viconf_pose.pose.position.y, 
                z=self.last_vicon.pose.position.z + curr_landpad_viconf_pose.pose.position.z
            )

            self.final_setpoint = setpoint

            # Publish current target setpoint and current pose to MAVROS
            if self.final_setpoint is not None:
                # Update timestamp on setpoint before publishing
                self.final_setpoint.header.stamp = self.get_clock().now().to_msg()
                self.setpoint_pub.publish(self.final_setpoint)
            # Publish the current pose (feedback) in vision_pose topic
            self.curr_pose_pub.publish(self.last_vicon)
        

def main(args=None):
    rclpy.init(args=args)
    node = CommNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down node.")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
