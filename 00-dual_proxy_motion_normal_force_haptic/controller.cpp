// This example tests the haptic device driver and the open-loop bilateral teleoperation controller.

#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "haptic_tasks/HapticController.h"
#include "../src/Logger.h"

#include <iostream>
#include <string>
#include <random>
#include <queue>

#define INIT            0
#define CONTROL         1

int state = INIT;

#include <signal.h>
bool runloop = false;
void sighandler(int){runloop = false;}

using namespace std;
using namespace Eigen;

// redis keys:
//// Haptic device related keys ////
// Maximum stiffness, damping and force specifications
vector<string> DEVICE_MAX_STIFFNESS_KEYS = {
	"sai2::ChaiHapticDevice::device0::specifications::max_stiffness",
	"sai2::ChaiHapticDevice::device1::specifications::max_stiffness",
	};
vector<string> DEVICE_MAX_DAMPING_KEYS = {
	"sai2::ChaiHapticDevice::device0::specifications::max_damping",
	"sai2::ChaiHapticDevice::device1::specifications::max_damping",
	};
vector<string> DEVICE_MAX_FORCE_KEYS = {
	"sai2::ChaiHapticDevice::device0::specifications::max_force",
	"sai2::ChaiHapticDevice::device1::specifications::max_force",
	};
// Set force and torque feedback of the haptic device
vector<string> DEVICE_COMMANDED_FORCE_KEYS = {
	"sai2::ChaiHapticDevice::device0::actuators::commanded_force",
	"sai2::ChaiHapticDevice::device1::actuators::commanded_force",
	};
vector<string> DEVICE_COMMANDED_TORQUE_KEYS = {
	"sai2::ChaiHapticDevice::device0::actuators::commanded_torque",
	"sai2::ChaiHapticDevice::device1::actuators::commanded_torque",
	};
vector<string> DEVICE_COMMANDED_GRIPPER_FORCE_KEYS = {
	"sai2::ChaiHapticDevice::device0::actuators::commanded_force_gripper",
	"sai2::ChaiHapticDevice::device1::actuators::commanded_force_gripper",
	};
// Haptic device current position and rotation
vector<string> DEVICE_POSITION_KEYS = {
	"sai2::ChaiHapticDevice::device0::sensors::current_position",
	"sai2::ChaiHapticDevice::device1::sensors::current_position",
	};
vector<string> DEVICE_ROTATION_KEYS = {
	"sai2::ChaiHapticDevice::device0::sensors::current_rotation",
	"sai2::ChaiHapticDevice::device1::sensors::current_rotation",
	};
vector<string> DEVICE_GRIPPER_POSITION_KEYS = {
	"sai2::ChaiHapticDevice::device0::sensors::current_position_gripper",
	"sai2::ChaiHapticDevice::device1::sensors::current_position_gripper",
	};
// Haptic device current velocity
vector<string> DEVICE_TRANS_VELOCITY_KEYS = {
	"sai2::ChaiHapticDevice::device0::sensors::current_trans_velocity",
	"sai2::ChaiHapticDevice::device1::sensors::current_trans_velocity",
	};
vector<string> DEVICE_ROT_VELOCITY_KEYS = {
	"sai2::ChaiHapticDevice::device0::sensors::current_rot_velocity",
	"sai2::ChaiHapticDevice::device1::sensors::current_rot_velocity",
	};
vector<string> DEVICE_GRIPPER_VELOCITY_KEYS = {
	"sai2::ChaiHapticDevice::device0::sensors::current_gripper_velocity",
	"sai2::ChaiHapticDevice::device1::sensors::current_gripper_velocity",
	};
vector<string> DEVICE_SENSED_FORCE_KEYS = {
	"sai2::ChaiHapticDevice::device0::sensors::sensed_force",
	"sai2::ChaiHapticDevice::device1::sensors::sensed_force",
	};
vector<string> DEVICE_SENSED_TORQUE_KEYS = {
	"sai2::ChaiHapticDevice::device0::sensors::sensed_torque",
	"sai2::ChaiHapticDevice::device1::sensors::sensed_torque",
	};

string DEVICE_GRIPPER_STATE_KEY = "gripper_state";

// dual proxy
string ROBOT_PROXY_KEY = "sai2::PegInsertion::00::dual_proxy::robot_proxy";
string HAPTIC_PROXY_KEY = "sai2::PegInsertion::00::dual_proxy::haptic_proxy";
string FORCE_SPACE_DIMENSION_KEY = "sai2::PegInsertion::00::dual_proxy::force_space_dimension";
string SIGMA_FORCE_KEY = "sai2::PegInsertion::00::dual_proxy::sigma_force";
string ROBOT_PROXY_ROT_KEY = "sai2::PegInsertion::00::dual_proxy::robot_proxy_rot";
string ROBOT_DEFAULT_ROT_KEY = "sai2::PegInsertion::00::dual_proxy::robot_default_rot";
string ROBOT_DEFAULT_POS_KEY = "sai2::PegInsertion::00::dual_proxy::robot_default_pos";

string HAPTIC_DEVICE_READY_KEY = "sai2::PegInsertion::00::dual_proxy::haptic_device_ready";
string HAPTIC_CONTROLLER_RUNNING_KEY = "sai2::PegInsertion::00::dual_proxy::haptic_controller_running";
string CONTROLLER_RUNNING_KEY = "sai2::PegInsertion::00::dual_proxy::controller_running";

int haptic_device_ready = 0;
int force_space_dimension = 0;
Matrix3d sigma_force = Matrix3d::Zero();
Vector3d haptic_proxy = Vector3d::Zero();
Vector3d robot_proxy = Vector3d::Zero();
Matrix3d robot_proxy_rot = Matrix3d::Identity();

int delayed_force_space_dimension = 0;
Matrix3d delayed_sigma_force = Matrix3d::Zero();
Vector3d delayed_haptic_proxy = Vector3d::Zero();
Vector3d delayed_robot_proxy = Vector3d::Zero();
Matrix3d delayed_robot_proxy_rot = Matrix3d::Identity();

RedisClient redis_client_local;
RedisClient redis_client_remote;

// communication function prototype
void communication(int delay);

// clamp function prototype
double clamp(double x, double upper, double lower);

int main(int argc, char* argv[]) {

	int communication_delay = 0;
	int device_z_rot = 0;
	if(argc >= 2) {
		communication_delay = stoi(argv[1]);
		if(argc == 3) {
			device_z_rot = stoi(argv[2]); // angle to rotate device frame about Z IN DEGREES
		}
	}

	// start redis clients
	redis_client_local = RedisClient();
	redis_client_local.connect();

	string remote_ip = "127.0.0.1";      // local
    // string remote_ip = "10.0.0.231";     // borns
    // string remote_ip = "10.9.8.1";
	int remote_port = 6379;
	redis_client_remote = RedisClient();
	redis_client_remote.connect(remote_ip, remote_port);

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// dual proxy parameters
	// double k_vir = 100.0;
	double k_vir = 200.0;
	double max_force_diff = 0.1;
	double max_force = 20.0;

	Vector3d desired_force = Vector3d::Zero();
	Vector3d prev_desired_force = Vector3d::Zero();
	Vector3d linear_centering_force = Vector3d::Zero();
	Vector3d linear_damping_force = Vector3d::Zero();
	Vector3d rotational_damping_moment = Vector3d::Zero();

	// haptic control
	//// Haptic teleoperation controller ////
	if(redis_client_remote.get(CONTROLLER_RUNNING_KEY) != "1") {
		std::cout << "run the robot controller before the haptic controller" << endl;
		return 0;
	}
	Vector3d robot_workspace_center = redis_client_remote.getEigenMatrixJSON(ROBOT_DEFAULT_POS_KEY);
	Matrix3d robot_rotation_default = redis_client_remote.getEigenMatrixJSON(ROBOT_DEFAULT_ROT_KEY);
	Matrix3d R_device_robot = Matrix3d::Identity();
	R_device_robot *= AngleAxisd(device_z_rot * M_PI/180.0, Vector3d::UnitZ()).toRotationMatrix(); // rotate frame by input Z angle
	Matrix3d device_rot_center = Matrix3d::Identity();
	Vector3d device_pos_center = Vector3d::Zero();
	Matrix3d device_release_rot = Matrix3d::Identity();
	auto teleop_task = new Sai2Primitives::HapticController(robot_workspace_center, robot_rotation_default, R_device_robot);
	teleop_task->_send_haptic_feedback = false;

	//User switch states
	teleop_task->UseGripperAsSwitch();
	int gripper_state = 0;
	int gripper_state_prev = gripper_state;

	robot_proxy = robot_workspace_center;
	robot_proxy_rot = robot_rotation_default;

	 //Task scaling factors
	// double Ks = 2.5;
	double Ks = 2.2;
	double KsR = 0.8;
	teleop_task->setScalingFactors(Ks, KsR);

	VectorXd _max_stiffness_device0 = redis_client_local.getEigenMatrixJSON(DEVICE_MAX_STIFFNESS_KEYS[0]);
	VectorXd _max_damping_device0 = redis_client_local.getEigenMatrixJSON(DEVICE_MAX_DAMPING_KEYS[0]);
	VectorXd _max_force_device0 = redis_client_local.getEigenMatrixJSON(DEVICE_MAX_FORCE_KEYS[0]);

	//set the device specifications to the haptic controller
	teleop_task->_max_linear_stiffness_device = _max_stiffness_device0[0];
	teleop_task->_max_angular_stiffness_device = _max_stiffness_device0[1];
	teleop_task->_max_linear_damping_device = _max_damping_device0[0];
	teleop_task->_max_angular_damping_device = _max_damping_device0[1];
	teleop_task->_max_force_device = _max_force_device0[0];
	teleop_task->_max_torque_device = _max_force_device0[1];

	double kp_haptic_linear = 5e-3 * _max_stiffness_device0[0]; // used for returning device to center
	double kv_haptic_linear = 0.9 * _max_damping_device0[0];
	// double kv_haptic = 0.5 * _max_damping_device0[0];
	double kv_haptic_angular = 0.5 * _max_damping_device0[1];

	Vector3d proxy_position_device_frame = Vector3d::Zero();

	// setup redis keys to be updated with the callback
	redis_client_local.createReadCallback(0);
	redis_client_local.createWriteCallback(0);

	// Objects to read from redis
    redis_client_local.addEigenToReadCallback(0, DEVICE_POSITION_KEYS[0], teleop_task->_current_position_device);
    redis_client_local.addEigenToReadCallback(0, DEVICE_ROTATION_KEYS[0], teleop_task->_current_rotation_device);
    redis_client_local.addEigenToReadCallback(0, DEVICE_TRANS_VELOCITY_KEYS[0], teleop_task->_current_trans_velocity_device);
    redis_client_local.addEigenToReadCallback(0, DEVICE_ROT_VELOCITY_KEYS[0], teleop_task->_current_rot_velocity_device);
    redis_client_local.addEigenToReadCallback(0, DEVICE_SENSED_FORCE_KEYS[0], teleop_task->_sensed_force_device);
    redis_client_local.addEigenToReadCallback(0, DEVICE_SENSED_TORQUE_KEYS[0], teleop_task->_sensed_torque_device);
    redis_client_local.addDoubleToReadCallback(0, DEVICE_GRIPPER_POSITION_KEYS[0], teleop_task->_current_position_gripper_device);
    redis_client_local.addDoubleToReadCallback(0, DEVICE_GRIPPER_VELOCITY_KEYS[0], teleop_task->_current_gripper_velocity_device);

	// redis_client_local.set(DEVICE_GRIPPER_STATE_KEY, "0");
	// redis_client_local.addIntToReadCallback(0, DEVICE_GRIPPER_STATE_KEY, gripper_state);

	// Objects to write to redis
	//write haptic commands
	redis_client_local.addEigenToWriteCallback(0, DEVICE_COMMANDED_FORCE_KEYS[0], teleop_task->_commanded_force_device);
	redis_client_local.addEigenToWriteCallback(0, DEVICE_COMMANDED_TORQUE_KEYS[0], teleop_task->_commanded_torque_device);
	redis_client_local.addDoubleToWriteCallback(0, DEVICE_COMMANDED_GRIPPER_FORCE_KEYS[0], teleop_task->_commanded_gripper_force_device);

	// create a timer
	double control_loop_freq = 1000.0;
	unsigned long long controller_counter = 0;
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(control_loop_freq); //Compiler en mode release
	double current_time = 0;
	double prev_time = 0;
	// double dt = 0;
	bool fTimerDidSleep = true;

	// setup data logging
	string folder = "../../00-dual_proxy_motion_normal_force_haptic/data_logging/data/";
	string filename = "data";
    auto logger = new Logging::Logger(10000, folder + filename);

	VectorXd log_haptic_time(1);
	log_haptic_time(0) = current_time;
	Vector3d log_haptic_position = Vector3d::Zero();
	Vector3d log_haptic_linear_velocity = Vector3d::Zero();
	Vector3d log_haptic_angular_velocity = Vector3d::Zero();
	Vector3d log_haptic_proxy = haptic_proxy;
	Vector3d log_haptic_commanded_force = Vector3d::Zero();
	Vector3d log_haptic_desired_force = Vector3d::Zero();
	Vector3d log_haptic_linear_centering_force = Vector3d::Zero();
	Vector3d log_haptic_linear_damping_force = Vector3d::Zero();
	Vector3d log_haptic_rotational_damping_moment = Vector3d::Zero();

	VectorXd log_haptic_orientation(Map<VectorXd>(device_rot_center.data(), device_rot_center.size()));
	
	logger->addVectorToLog(&log_haptic_time, "haptic_time");
	logger->addVectorToLog(&log_haptic_position, "haptic_position");
	logger->addVectorToLog(&log_haptic_linear_velocity, "haptic_linear_velocity");
	logger->addVectorToLog(&log_haptic_angular_velocity, "haptic_angular_velocity");
	logger->addVectorToLog(&log_haptic_proxy, "haptic_proxy");
	logger->addVectorToLog(&log_haptic_commanded_force, "haptic_command_force");
	logger->addVectorToLog(&log_haptic_desired_force, "haptic_desired_force");
	logger->addVectorToLog(&log_haptic_linear_centering_force, "haptic_linear_centering_force");
	logger->addVectorToLog(&log_haptic_linear_damping_force, "haptic_linear_damping_force");
	logger->addVectorToLog(&log_haptic_rotational_damping_moment, "haptic_rotational_damping_moment");
	logger->addVectorToLog(&log_haptic_orientation, "haptic_orientation");

	redis_client_remote.set(HAPTIC_CONTROLLER_RUNNING_KEY, "1");
	logger->start();

	// start communication thread and loop
	runloop = true;
	thread communication_thread(communication, communication_delay);

	// start timer
	double start_time = timer.elapsedTime(); //secs

	while (runloop) {
		// wait for next scheduled loop
		timer.waitForNextLoop();
		current_time = timer.elapsedTime() - start_time;

		// read haptic state and robot state
		gripper_state_prev = gripper_state;
		redis_client_local.executeReadCallback(0);

		teleop_task->UseGripperAsSwitch();
		gripper_state = teleop_task->gripper_state;

		if(state == INIT) {
  			// compute homing haptic device
  			teleop_task->HomingTask();

			if(teleop_task->device_homed && gripper_state) {
  			// cout << teleop_task->device_homed << endl;

			// if(controller_counter > 3000 && gripper_state) {
				// teleop_task->device_homed = true;

				// teleop_task->setRobotCenter(haptic_proxy, robot_rotation_default);
				teleop_task->setDeviceCenter(teleop_task->_current_position_device, teleop_task->_current_rotation_device);
				device_rot_center = teleop_task->_current_rotation_device;
				device_pos_center = teleop_task->_current_position_device;

				// Reinitialize controllers
				teleop_task->reInitializeTask();

				haptic_device_ready = 1;
				state = CONTROL;
			}
		}

		else if(state == CONTROL) {
			// compute haptic commands
			if(gripper_state) { // full control
				if(!gripper_state_prev) {
					device_rot_center = device_release_rot.transpose() * teleop_task->_current_rotation_device;
					teleop_task->setDeviceCenter(device_pos_center, device_rot_center);
				}
				teleop_task->computeHapticCommands6d(robot_proxy, robot_proxy_rot);
			}
			else { // only position control
				if(gripper_state_prev) {
					device_release_rot = teleop_task->_current_rotation_device * device_rot_center.transpose();
				}
				teleop_task->computeHapticCommands3d(robot_proxy);
			}

			Vector3d device_position = teleop_task->_current_position_device;
			proxy_position_device_frame = teleop_task->_home_position_device + teleop_task->_Rotation_Matrix_DeviceToRobot * (delayed_haptic_proxy - teleop_task->_center_position_robot) / Ks;

			desired_force = k_vir * delayed_sigma_force * (proxy_position_device_frame - device_position);
			// ensure desired force has no "pulling" component
			// TODO: use robot orientation for probe axis
			if(desired_force.z() < 0.0) desired_force.z() = 0;
			// if((desired_force.z() - prev_desired_force.z()) < 0) desired_force.z() = 0;
			Vector3d desired_force_diff = desired_force - prev_desired_force;
			if(desired_force_diff.norm() > max_force_diff) {
				desired_force = prev_desired_force + max_force_diff * desired_force_diff/desired_force_diff.norm();
			}
			if(desired_force.norm() > max_force) {
				desired_force *= max_force/desired_force.norm();
			}

			// send force feedback to haptic device
			teleop_task->_commanded_force_device = desired_force;
			
			// add haptic centering force
			linear_centering_force = - kp_haptic_linear * (teleop_task->_current_position_device - teleop_task->_home_position_device);
			teleop_task->_commanded_force_device += linear_centering_force;

			// add linear damping
			// linear_damping_force = - kv_haptic_linear * delayed_sigma_force * teleop_task->_current_trans_velocity_device;
			Vector3d vel = teleop_task->_current_trans_velocity_device;
			double speed = vel.norm();
			// std::cout << "speed: " << speed << std::endl << std::endl;
			double kv = kv_haptic_linear * clamp((speed - 0.0)/(0.01 - 0.0), 0.0, 1.0);
			// std::cout << "kv_haptic_linear: " << kv_haptic_linear << std::endl << std::endl;
			// std::cout << "kv: " << kv << std::endl << std::endl;
			// if(teleop_task->_current_trans_velocity_device.norm() < 0.1) kv = 0;
			linear_damping_force = - kv_haptic_linear * delayed_sigma_force * teleop_task->_current_trans_velocity_device;
			// linear_damping_force = - kv * delayed_sigma_force * teleop_task->_current_trans_velocity_device;
			teleop_task->_commanded_force_device += linear_damping_force;

			// add rotational damping for sigma haptic device
			rotational_damping_moment = - kv_haptic_angular * teleop_task->_current_rot_velocity_device;
			teleop_task->_commanded_torque_device = rotational_damping_moment;

			// remember values 
			prev_desired_force = desired_force;
		}

		// write control torques
		redis_client_local.executeWriteCallback(0);

		// update logger variables
		log_haptic_time(0) = current_time;
		log_haptic_position = teleop_task->_current_position_device;
		log_haptic_linear_velocity = teleop_task->_current_trans_velocity_device;
		log_haptic_angular_velocity = teleop_task->_current_rot_velocity_device;
		log_haptic_proxy = proxy_position_device_frame;
		log_haptic_commanded_force = teleop_task->_commanded_force_device;
		log_haptic_desired_force = desired_force;
		log_haptic_linear_centering_force = linear_centering_force;
		log_haptic_linear_damping_force = linear_damping_force;
		log_haptic_rotational_damping_moment = rotational_damping_moment;
		log_haptic_orientation = Map<VectorXd>(teleop_task->_current_rotation_device.data(), teleop_task->_current_rotation_device.size());
		
		controller_counter++;
	}

	logger->stop();
	communication_thread.join();

	//// Send zero force/torque to haptic device through Redis keys ////
	redis_client_local.setEigenMatrixJSON(DEVICE_COMMANDED_FORCE_KEYS[0], Vector3d::Zero());
	redis_client_local.setEigenMatrixJSON(DEVICE_COMMANDED_TORQUE_KEYS[0], Vector3d::Zero());
	redis_client_local.set(DEVICE_COMMANDED_GRIPPER_FORCE_KEYS[0], "0.0");

	double end_time = timer.elapsedTime();
	std::cout << "\n";
	std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
	std::cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}



void communication(int delay) {
	// prepare delay
	const double communication_delay_ms = delay;

	queue<Vector3d> robot_proxy_buffer;
	queue<Matrix3d> robot_proxy_rot_buffer;
	queue<Vector3d> haptic_proxy_buffer;
	queue<Matrix3d> sigma_force_buffer;
	queue<int> force_space_dimension_buffer;

	// prepare redis client remote communication
	redis_client_remote.createReadCallback(0);
	redis_client_remote.createWriteCallback(0);

	redis_client_remote.addEigenToReadCallback(0, HAPTIC_PROXY_KEY, haptic_proxy);
	redis_client_remote.addIntToReadCallback(0, FORCE_SPACE_DIMENSION_KEY, force_space_dimension);
	redis_client_remote.addEigenToReadCallback(0, SIGMA_FORCE_KEY, sigma_force);

	redis_client_remote.addEigenToWriteCallback(0, ROBOT_PROXY_KEY, delayed_robot_proxy);
	redis_client_remote.addEigenToWriteCallback(0, ROBOT_PROXY_ROT_KEY, delayed_robot_proxy_rot);
	redis_client_remote.addIntToWriteCallback(0, HAPTIC_DEVICE_READY_KEY, haptic_device_ready);

	// create a timer
	double communication_freq = 50.0;
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(communication_freq); //Compiler en mode release
	double current_time = 0;
	double prev_time = 0;
	// double dt = 0;
	bool fTimerDidSleep = true;
	double start_time = timer.elapsedTime(); //secs

	unsigned long long communication_counter = 0;
	const int communication_delay_ncycles = communication_delay_ms / 1000.0 * communication_freq;
	
	while(runloop) {
		timer.waitForNextLoop();

		redis_client_remote.executeReadCallback(0);

		if(communication_delay_ncycles == 0) {
			delayed_haptic_proxy = haptic_proxy;
			delayed_robot_proxy = robot_proxy;
			delayed_robot_proxy_rot = robot_proxy_rot;
			delayed_sigma_force = sigma_force;
			delayed_force_space_dimension = force_space_dimension;
		}
		else {
			robot_proxy_buffer.push(robot_proxy);
			robot_proxy_rot_buffer.push(robot_proxy_rot);
			haptic_proxy_buffer.push(haptic_proxy);
			sigma_force_buffer.push(sigma_force);
			force_space_dimension_buffer.push(force_space_dimension);

			if(communication_counter > communication_delay_ncycles) {
				delayed_haptic_proxy = haptic_proxy_buffer.front();
				delayed_robot_proxy = robot_proxy_buffer.front();
				delayed_robot_proxy_rot = robot_proxy_rot_buffer.front();
				delayed_sigma_force = sigma_force_buffer.front();
				delayed_force_space_dimension = force_space_dimension_buffer.front();

				robot_proxy_buffer.pop();
				robot_proxy_rot_buffer.pop();
				haptic_proxy_buffer.pop();
				sigma_force_buffer.pop();
				force_space_dimension_buffer.pop();

				communication_counter--;
			}
		}

		redis_client_remote.executeWriteCallback(0);

		communication_counter++;
	}

	redis_client_remote.set(HAPTIC_DEVICE_READY_KEY, "0");
	redis_client_remote.set(HAPTIC_CONTROLLER_RUNNING_KEY, "0");

	double end_time = timer.elapsedTime();
	std::cout << "\n";
	std::cout << "Communication Loop run time  : " << end_time << " seconds\n";
	std::cout << "Communication Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Communication Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}

double clamp(double x, double lower, double upper)
{
    return min(upper, max(x, lower));
}
