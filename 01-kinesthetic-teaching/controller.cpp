// Controller and Data Logger for kinesthetic teaching of the robot

#include "Sai2Model.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "tasks/JointTask.h"
#include "tasks/PosOriTask.h"
#include "filters/ButterworthFilter.h"
#include "../src/Logger.h"
#include "perception/ForceSpaceParticleFilter.h"

#include <iostream>
#include <string>
#include <random>
#include <queue>

#define INIT            0
#define CONTROL         1

#include <signal.h>
bool runloop = false;
void sighandler(int){runloop = false;}

using namespace std;
using namespace Eigen;

const string robot_file = "./resources/panda_arm.urdf";

// redis keys:
// robot local control loop
string JOINT_ANGLES_KEY = "sai2::PegInsertion::01::simviz::sensors::q";
string JOINT_VELOCITIES_KEY = "sai2::PegInsertion::01::simviz::sensors::dq";
string ROBOT_COMMAND_TORQUES_KEY = "sai2::PegInsertion::01::simviz::actuators::tau_cmd";

string ROBOT_SENSED_FORCE_KEY = "sai2::PegInsertion::01::simviz::sensors::sensed_force";

string MASSMATRIX_KEY;
string CORIOLIS_KEY;

// posori task state information
string ROBOT_EE_POS_KEY = "sai2::PegInsertion::01::robot::ee_pos";
string ROBOT_EE_ORI_KEY = "sai2::PegInsertion::01::robot::ee_ori";
string ROBOT_EE_FORCE_KEY = "sai2::PegInsertion::01::robot::ee_force";
string ROBOT_EE_MOMENT_KEY = "sai2::PegInsertion::01::robot::ee_moment";

RedisClient redis_client;

// particle filter parameters
const int n_particles = 1000;
MatrixXd particle_positions_to_redis = MatrixXd::Zero(3, n_particles);
int force_space_dimension = 0;
int prev_force_space_dimension = 0;
Matrix3d sigma_force = Matrix3d::Identity();
Matrix3d sigma_motion = Matrix3d::Identity();

Vector3d motion_control_pfilter;
Vector3d force_control_pfilter;
Vector3d measured_velocity_pfilter;
Vector3d measured_force_pfilter;

queue<Vector3d> pfilter_motion_control_buffer;
queue<Vector3d> pfilter_force_control_buffer;
queue<Vector3d> pfilter_sensed_force_buffer;
queue<Vector3d> pfilter_sensed_velocity_buffer;

const double control_loop_freq = 1000.0;
const double pfilter_freq = 50.0;
const double freq_ratio_filter_control = pfilter_freq / control_loop_freq;

// set control link and point for posori task
// const string link_name = "end_effector";
// const Vector3d pos_in_link = Vector3d(0.0,0.0,0.035);

// set sensor frame transform in end-effector frame
Affine3d sensor_transform_in_link = Affine3d::Identity();
const Vector3d sensor_pos_in_link = Eigen::Vector3d(0.0,0.0,0.0406);

// particle filter loop
void particle_filter();

const bool flag_simulation = false;
// const bool flag_simulation = true;

int main() {

	if(!flag_simulation) {
		ROBOT_COMMAND_TORQUES_KEY = "sai2::FrankaPanda::Bonnie::actuators::fgc";
		JOINT_ANGLES_KEY  = "sai2::FrankaPanda::Bonnie::sensors::q";
		JOINT_VELOCITIES_KEY = "sai2::FrankaPanda::Bonnie::sensors::dq";
		MASSMATRIX_KEY = "sai2::FrankaPanda::Bonnie::sensors::model::massmatrix";
		CORIOLIS_KEY = "sai2::FrankaPanda::Bonnie::sensors::model::coriolis";
		ROBOT_SENSED_FORCE_KEY = "sai2::ATIGamma_Sensor::Bonnie::force_torque";
	}

	// start redis client local
	redis_client = RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load robots
	Affine3d T_world_robot = Affine3d::Identity();
	T_world_robot.translation() = Vector3d(0, 0, 0);
	auto robot = new Sai2Model::Sai2Model(robot_file, false, T_world_robot);

	robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
	robot->updateModel();

	int dof = robot->dof();
	VectorXd command_torques = VectorXd::Zero(dof);
	VectorXd coriolis = VectorXd::Zero(dof);
	int state = INIT;
	MatrixXd N_prec = MatrixXd::Identity(dof,dof);

	// joint task
	auto joint_task = new Sai2Primitives::JointTask(robot);
	VectorXd joint_task_torques = VectorXd::Zero(dof);
	joint_task->_use_interpolation_flag = false;
	joint_task->_use_velocity_saturation_flag = true;

	joint_task->_kp = 200.0;
	joint_task->_kv = 25.0;
	joint_task->_ki = 50.0;

	// Set initial starting configuration for teaching
    // joint_task->_desired_position << -0.137034, 0.196574, 0.0870424, -2.26689, -0.0722293, 2.46444, 0.0702875; 
	// joint_task->_desired_position << -0.0872258, 0.0112011, -0.0211492, -2.63778, -0.276039, 3.13521, -0.714044; // 12-02-23 Data Collection Position
	joint_task->_desired_position = robot->_q; // use current robot config as init config

	// PosOri task
	const string link_name = "end_effector";
	// const Vector3d pos_in_link = Vector3d(0, 0, 0.10); // distance to tip-of-peg mounted on force sensor
	const Vector3d pos_in_link = Vector3d(0, 0, 0.169); // distance to center of stock panda gripper contact patches mounted on force sensor
	auto posori_task = new Sai2Primitives::PosOriTask(robot, link_name, pos_in_link);
	Vector3d x_init = posori_task->_current_position;
	Matrix3d R_init = posori_task->_current_orientation;
	Matrix3d R_default = Matrix3d::Identity();
	Vector3d pos_default = Vector3d::Zero();
	robot->rotation(R_default, link_name);
	robot->position(pos_default, link_name, pos_in_link);

	// Set Desired Peg Position
	posori_task->_desired_position << 0.499218,0.034793,0.094611;
	posori_task->_desired_orientation << 0.999998,-0.001182,-0.001852,
										 -0.001151,-0.999856,0.016908,
										 -0.001872,-0.016906,-0.999855;

	VectorXd posori_task_torques = VectorXd::Zero(dof);
	posori_task->_use_interpolation_flag = false;

	posori_task->_otg->setMaxLinearVelocity(0.30);
	posori_task->_otg->setMaxLinearAcceleration(1.0);
	posori_task->_otg->setMaxLinearJerk(5.0);

	posori_task->_otg->setMaxAngularVelocity(M_PI/1.5);
	posori_task->_otg->setMaxAngularAcceleration(3*M_PI);
	posori_task->_otg->setMaxAngularJerk(15*M_PI);

	posori_task->_kp_pos = 100.0;
	posori_task->_kv_pos = 17.0;

	posori_task->_kp_ori = 200.0;
	posori_task->_kv_ori = 23.0;

	posori_task->_use_velocity_saturation_flag = true;
	posori_task->_linear_saturation_velocity = 0.3;
	posori_task->_angular_saturation_velocity = M_PI/2.0;

	// use lambda smoothing instead of lambda truncation
	posori_task->_use_lambda_truncation_flag = false;
	posori_task->_e_sing = 1e-1;
	posori_task->_e_max = 1e-1;  // bounds subject to tuning
	posori_task->_e_min = 5e-2;
	
	Vector3d ee_pos = Vector3d::Zero();
	Vector3d ee_linear_vel = Vector3d::Zero();
	Vector3d ee_angular_vel = Vector3d::Zero();
	Matrix3d ee_ori = Matrix3d::Identity();

	// force sensing
	Matrix3d R_link_sensor = Matrix3d::Identity();
	sensor_transform_in_link.translation() = sensor_pos_in_link;
	sensor_transform_in_link.linear() = R_link_sensor;

	VectorXd sensed_force_moment_local_frame = VectorXd::Zero(6);
	VectorXd sensed_force_moment_world_frame = VectorXd::Zero(6);
	VectorXd force_bias = VectorXd::Zero(6);
	double tool_mass = 0.023;
	Vector3d tool_com = Vector3d::Zero();

	Vector3d init_force = Vector3d::Zero();
	bool first_loop = true;

	// remove inertial forces from tool
	Vector3d tool_velocity = Vector3d::Zero();
	Vector3d prev_tool_velocity = Vector3d::Zero();
	Vector3d tool_acceleration = Vector3d::Zero();
	Vector3d tool_inertial_forces = Vector3d::Zero();

	// setup redis keys to be updated with the callback
	redis_client.createReadCallback(0);
	redis_client.createWriteCallback(0);

	// Objects to read from redis
    redis_client.addEigenToReadCallback(0, JOINT_ANGLES_KEY, robot->_q);
    redis_client.addEigenToReadCallback(0, JOINT_VELOCITIES_KEY, robot->_dq);

    MatrixXd mass_from_robot = MatrixXd::Identity(dof,dof);
    VectorXd coriolis_from_robot = VectorXd::Zero(dof);
	if(!flag_simulation) {
		redis_client.addEigenToReadCallback(0, MASSMATRIX_KEY, mass_from_robot);
		redis_client.addEigenToReadCallback(0, CORIOLIS_KEY, coriolis_from_robot);
	}

    redis_client.addEigenToReadCallback(0, ROBOT_SENSED_FORCE_KEY, sensed_force_moment_local_frame);

	// Objects to write to redis
	redis_client.addEigenToWriteCallback(0, ROBOT_COMMAND_TORQUES_KEY, command_torques);

	// write internal controller state to redis
    redis_client.addEigenToWriteCallback(0, ROBOT_EE_POS_KEY, posori_task->_current_position);
    redis_client.addEigenToWriteCallback(0, ROBOT_EE_ORI_KEY, posori_task->_current_orientation);

	// create a timer
	unsigned long long controller_counter = 0;
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(control_loop_freq); //Compiler en mode release
	double current_time = 0;
	double prev_time = 0;
	double dt = 0;
	bool fTimerDidSleep = true;

	double time_until_float = 0;

	// setup data logging
	string folder = "../../01-kinesthetic-teaching/data_logging/data/";
	string filename = "data";
    auto logger = new Logging::Logger(10000, folder + filename);
	
	VectorXd log_robot_time(1);
	Vector3d log_robot_ee_pos_error = Vector3d::Zero();
	Vector3d log_robot_ee_ori_error = Vector3d::Zero();
	VectorXd log_sensed_force_moments = VectorXd::Zero(6);
	VectorXd log_robot_ee_orientation = VectorXd::Zero(9);
	Vector3d log_robot_ee_position = Vector3d::Zero();
	Vector3d log_robot_ee_linear_velocity = Vector3d::Zero();
	Vector3d log_robot_ee_angular_velocity = Vector3d::Zero();
	VectorXd log_joint_angles = robot->_q;
	VectorXd log_joint_velocities = robot->_dq;

	logger->addVectorToLog(&log_robot_time, "robot_time");
	logger->addVectorToLog(&log_robot_ee_pos_error, "robot_ee_pos_error");
	logger->addVectorToLog(&log_robot_ee_ori_error, "robot_ee_ori_error");		
	logger->addVectorToLog(&log_sensed_force_moments, "sensed_forces_moments");
	logger->addVectorToLog(&log_robot_ee_position, "robot_ee_position");
	logger->addVectorToLog(&log_robot_ee_orientation, "robot_ee_orientation");
	logger->addVectorToLog(&log_robot_ee_linear_velocity, "robot_ee_linear_velocity");
	logger->addVectorToLog(&log_robot_ee_angular_velocity, "robot_ee_angular_velocity");
	logger->addVectorToLog(&log_joint_angles, "joint_angles");
	logger->addVectorToLog(&log_joint_velocities, "joint_velocities");

	bool printed_control_status = false;

	// // start particle filter thread
	// runloop = true;
	// redis_client.set(CONTROLLER_RUNNING_KEY,"1");
	// thread particle_filter_thread(particle_filter);

	// start timer
	std::cout << "starting robot controller" << endl;
	double start_time = timer.elapsedTime(); //secs
	runloop = true;

	time_until_float = 0.0;

	while (runloop) {
		// wait for next scheduled loop
		timer.waitForNextLoop();
		current_time = timer.elapsedTime() - start_time;
		dt = current_time - prev_time;

		// read robot state
		redis_client.executeReadCallback(0);
		if(flag_simulation) {
			robot->updateModel();
			robot->coriolisForce(coriolis);
		}
		else {
			robot->updateKinematics();
			robot->_M = mass_from_robot;
			robot->updateInverseInertia();
			coriolis = coriolis_from_robot;
		}

		N_prec.setIdentity(dof,dof);

		posori_task->updateTaskModel(N_prec);
		N_prec = posori_task->_N;

		joint_task->updateTaskModel(N_prec);

		// add bias and ee weight to sensed forces
		sensed_force_moment_local_frame -= force_bias;
		Matrix3d R_world_sensor;
		robot->rotation(R_world_sensor, link_name);
		R_world_sensor = R_world_sensor * R_link_sensor;
		Vector3d p_tool_local_frame = tool_mass * R_world_sensor.transpose() * Vector3d(0,0,-9.81);
		sensed_force_moment_local_frame.head(3) += p_tool_local_frame;
		sensed_force_moment_local_frame.tail(3) += tool_com.cross(p_tool_local_frame);

		if(first_loop) {
			init_force = sensed_force_moment_local_frame.head(3);
			std::cout << "moving to initial position" << endl;
			first_loop = false;
		}
		sensed_force_moment_local_frame.head(3) -= init_force;

		// update forces for posori task
		posori_task->updateSensedForceAndMoment(sensed_force_moment_local_frame.head(3), sensed_force_moment_local_frame.tail(3));
		sensed_force_moment_world_frame.head(3) = R_world_sensor * sensed_force_moment_local_frame.head(3);
		sensed_force_moment_world_frame.tail(3) = R_world_sensor * sensed_force_moment_local_frame.tail(3);

		if(state == INIT) {
			printed_control_status = false;
			joint_task->updateTaskModel(MatrixXd::Identity(dof,dof));

			joint_task->computeTorques(joint_task_torques);
			command_torques = joint_task_torques + coriolis;

			if((joint_task->_desired_position - joint_task->_current_position).norm() < 0.1) {

				time_until_float +=  dt;
				// std::cout << std::to_string(time_until_float) << endl;

				if (time_until_float > 10)
				{
					state = CONTROL;
				}
			}
		}

		else if(state == CONTROL) {
			
			// Reinitialize controllers
			joint_task->reInitializeTask();

			joint_task->_kp = 50.0;
			joint_task->_kv = 13.0;
			joint_task->_ki = 0.0;

			if(!printed_control_status){
				// Start logging 
				logger->start();
				std::cout << "starting robot logger" << endl;
				std::cout << "begin kinesthetic teaching" << endl;
				printed_control_status = true;
			}

			try	{
				posori_task->computeTorques(posori_task_torques);
			}
			catch(exception e) {
				cout << "control cycle: " << controller_counter << endl;
				cout << "error in the torque computation of posori_task:" << endl;
				cerr << e.what() << endl;
				cout << "setting torques to zero for this control cycle" << endl;
				cout << endl;
				// posori_task_torques.setZero(); // set task torques to zero, TODO: test this
			}

			// command_torques = coriolis;	
			command_torques.setZero();
		}

		// write control
		redis_client.executeWriteCallback(0);

		// particle filter
		// pfilter_motion_control_buffer.push(sigma_motion * (robot_proxy - posori_task->_current_position) * freq_ratio_filter_control);
		// pfilter_force_control_buffer.push(sigma_force * (robot_proxy - posori_task->_current_position) * freq_ratio_filter_control);
		// // pfilter_motion_control_buffer.push(sigma_motion * posori_task->_Lambda_modified.block<3,3>(0,0) * posori_task->_linear_motion_control * freq_ratio_filter_control);
		// // pfilter_force_control_buffer.push(sigma_force * posori_task->_linear_force_control * freq_ratio_filter_control);

		// pfilter_sensed_velocity_buffer.push(posori_task->_current_velocity * freq_ratio_filter_control);
		// pfilter_sensed_force_buffer.push(sensed_force_moment_world_frame.head(3) * freq_ratio_filter_control);

		// motion_control_pfilter += pfilter_motion_control_buffer.back();
		// force_control_pfilter += pfilter_force_control_buffer.back();
		// measured_velocity_pfilter += pfilter_sensed_velocity_buffer.back();
		// measured_force_pfilter += pfilter_sensed_force_buffer.back();

		// if(pfilter_motion_control_buffer.size() > 1/freq_ratio_filter_control) {
		// 	motion_control_pfilter -= pfilter_motion_control_buffer.front();
		// 	force_control_pfilter -= pfilter_force_control_buffer.front();
		// 	measured_velocity_pfilter -= pfilter_sensed_velocity_buffer.front();
		// 	measured_force_pfilter -= pfilter_sensed_force_buffer.front();

		// 	pfilter_motion_control_buffer.pop();
		// 	pfilter_force_control_buffer.pop();
		// 	pfilter_sensed_velocity_buffer.pop();
		// 	pfilter_sensed_force_buffer.pop();
		// }

		// update logger values
		robot->position(ee_pos, link_name, pos_in_link);
		robot->linearVelocity(ee_linear_vel, link_name, pos_in_link);
		robot->angularVelocity(ee_angular_vel, link_name, pos_in_link);
		robot->rotation(ee_ori, link_name);

		log_robot_time(0) = current_time;
		log_robot_ee_pos_error = ee_pos - posori_task->_desired_position;
		log_robot_ee_ori_error = posori_task->_orientation_error;
		log_sensed_force_moments = sensed_force_moment_local_frame;	
		log_robot_ee_position = ee_pos;
		log_robot_ee_orientation = Map<VectorXd>(ee_ori.data(), ee_ori.size());
		log_robot_ee_linear_velocity = ee_linear_vel;
		log_robot_ee_angular_velocity = ee_angular_vel;
		log_joint_angles = robot->_q;
		log_joint_velocities = robot->_dq;		


		prev_time = current_time;
		controller_counter++;
	}

	// wait for particle filter thread
	// particle_filter_thread.join();

	//// Send zero force/torque to robot ////
	command_torques.setZero();
	redis_client.setEigenMatrixJSON(ROBOT_COMMAND_TORQUES_KEY, command_torques);

	double end_time = timer.elapsedTime();
	std::cout << "\n";
	std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
	std::cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";

	// stop logger
	logger->stop();
}



void particle_filter() {
	// start redis client for particles
	auto redis_client_particles = RedisClient();
	redis_client_particles.connect();

	unsigned long long pf_counter = 0;

	// create particle filter
	auto pfilter = new Sai2Primitives::ForceSpaceParticleFilter(n_particles);

	pfilter->_mean_scatter = 0.0;
	pfilter->_std_scatter = 0.025;

	// pfilter->_alpha_add = 0.3;
	// pfilter->_alpha_remove = 0.05;

	// pfilter->_F_low = 2.0;
	// pfilter->_F_high = 6.0;
	// pfilter->_v_low = 0.02;
	// pfilter->_v_high = 0.07;

	// pfilter->_F_low_add = 5.0;
	// pfilter->_F_high_add = 10.0;
	// pfilter->_v_low_add = 0.02;
	// pfilter->_v_high_add = 0.1;

	pfilter->_alpha_add = 0.15;
	pfilter->_alpha_remove = 0.05;

	pfilter->_F_low = 1.0;
	pfilter->_F_high = 6.0;
	pfilter->_v_low = 0.05;
	pfilter->_v_high = 1.0;

	pfilter->_F_low_add = 2.0;
	pfilter->_F_high_add = 10.0;
	pfilter->_v_low_add = 0.05;
	pfilter->_v_high_add = 1.0;

	Vector3d evals = Vector3d::Zero();
	Matrix3d evecs = Matrix3d::Identity();

	// create a timer
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(pfilter_freq); //Compiler en mode release
	double current_time = 0;
	double prev_time = 0;
	// double dt = 0;
	bool fTimerDidSleep = true;
	double start_time = timer.elapsedTime(); //secs

	while(runloop) {
		timer.waitForNextLoop();

		pfilter->update(motion_control_pfilter, force_control_pfilter, measured_velocity_pfilter, measured_force_pfilter);
		sigma_force = pfilter->getSigmaForce();
		sigma_motion = Matrix3d::Identity() - sigma_force;
		force_space_dimension = pfilter->_force_space_dimension;

		// for(int i=0 ; i<n_particles ; i++)
		// {
		// 	particle_positions_to_redis.col(i) = pfilter->_particles[i];
		// }
		// redis_client_particles.setEigenMatrixJSON(PARTICLE_POSITIONS_KEY, particle_positions_to_redis);

		pf_counter++;
	}

	double end_time = timer.elapsedTime();
	std::cout << "\n";
	std::cout << "Particle Filter Loop run time  : " << end_time << " seconds\n";
	std::cout << "Particle Filter Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Particle Filter Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";
}
