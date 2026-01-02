#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>

#include "use-ikfom.hpp"
#include "esekfom.hpp"

/*
这个hpp主要包含：
/*
This hpp file mainly contains:
IMU data preprocessing: IMU initialization, IMU forward propagation, backward propagation compensating motion distortion
[这个hpp主要包含：
IMU数据预处理：IMU初始化，IMU正向传播，反向传播补偿运动失真]
*/

#define MAX_INI_COUNT (10)  //Maximum number of iterations[最大迭代次数]
//Determine the time order of points (note: curvature stores the timestamp)[判断点的时间先后顺序(注意curvature中存储的是时间戳)]
const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void set_param(const V3D &transl, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias);
  Eigen::Matrix<double, 12, 12> Q;    //Noise covariance matrix, corresponding to Q in equation (8) of the paper[噪声协方差矩阵  对应论文式(8)中的Q]
  void Process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI::Ptr &pcl_un_);

  V3D cov_acc;             //Acceleration covariance[加速度协方差]
  V3D cov_gyr;             //Angular velocity covariance[角速度协方差]
  V3D cov_acc_scale;       //External initial acceleration covariance[外部传入的 初始加速度协方差]
  V3D cov_gyr_scale;       //External initial angular velocity covariance[外部传入的 初始角速度协方差]
  V3D cov_bias_gyr;        //Angular velocity bias covariance[角速度bias的协方差]
  V3D cov_bias_acc;        //Acceleration bias covariance[加速度bias的协方差]
  double first_lidar_time; //Timestamp of first point cloud in current frame[当前帧第一个点云时间]

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;        //Current frame point cloud without distortion[当前帧点云未去畸变]
  sensor_msgs::ImuConstPtr last_imu_;     // Last frame IMU[上一帧imu]
  vector<Pose6D> IMUpose;                 // Store IMU pose (used for backward propagation)[存储imu位姿(反向传播用)]
  M3D Lidar_R_wrt_IMU;                    // Rotation extrinsic parameter from Lidar to IMU[lidar到IMU的旋转外参]
  V3D Lidar_T_wrt_IMU;                    // Translation extrinsic parameter from Lidar to IMU[lidar到IMU的平移外参]
  V3D mean_acc;                           //Acceleration mean, used for variance calculation[加速度均值,用于计算方差]
  V3D mean_gyr;                           //Angular velocity mean, used for variance calculation[角速度均值，用于计算方差]
  V3D angvel_last;                        //Last frame angular velocity[上一帧角速度]
  V3D acc_s_last;                         //Last frame acceleration[上一帧加速度]
  double start_timestamp_;                //Start timestamp[开始时间戳]
  double last_lidar_end_time_;            //Last frame end timestamp[上一帧结束时间戳]
  int init_iter_num = 1;                  //Initialization iteration count[初始化迭代次数]
  bool b_first_frame_ = true;             //Whether it is the first frame[是否是第一帧]
  bool imu_need_init_ = true;             //Whether IMU needs to be initialized[是否需要初始化imu]
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;                          //Initialization iteration count[初始化迭代次数]
  Q = process_noise_cov();                    //Call process_noise_cov in use-ikfom.hpp to initialize noise covariance[调用use-ikfom.hpp里面的process_noise_cov初始化噪声协方差]
  cov_acc = V3D(0.1, 0.1, 0.1);               //Initialize acceleration covariance[加速度协方差初始化]
  cov_gyr = V3D(0.1, 0.1, 0.1);               //Initialize angular velocity covariance[角速度协方差初始化]
  cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001); //Initialize angular velocity bias covariance[角速度bias协方差初始化]
  cov_bias_acc = V3D(0.0001, 0.0001, 0.0001); //Initialize acceleration bias covariance[加速度bias协方差初始化]
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;                       //Initialize last frame angular velocity[上一帧角速度初始化]
  Lidar_T_wrt_IMU = Zero3d;                   // Initialize Lidar to IMU translation extrinsic[lidar到IMU的位置外参初始化]
  Lidar_R_wrt_IMU = Eye3d;                    // Initialize Lidar to IMU rotation extrinsic[lidar到IMU的旋转外参初始化]
  last_imu_.reset(new sensor_msgs::Imu());    //Initialize last frame IMU[上一帧imu初始化]
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset()   //Reset parameters[重置参数]
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  imu_need_init_ = true;                   //Whether IMU needs to be initialized[是否需要初始化imu]
  start_timestamp_ = -1;                   //Start timestamp[开始时间戳]
  init_iter_num = 1;                       //Initialization iteration count[初始化迭代次数]
  IMUpose.clear();                         // Clear IMU poses[imu位姿清空]
  last_imu_.reset(new sensor_msgs::Imu()); //Initialize last frame IMU[上一帧imu初始化]
  cur_pcl_un_.reset(new PointCloudXYZI()); //Initialize current frame point cloud without distortion[当前帧点云未去畸变初始化]
}

//Pass in external parameters[传入外部参数]
void ImuProcess::set_param(const V3D &transl, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias)  
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
  cov_gyr_scale = gyr;
  cov_acc_scale = acc;
  cov_bias_gyr = gyr_bias;
  cov_bias_acc = acc_bias;
}


//IMU initialization: initialize state variable x using the average value of initial IMU frames[IMU初始化：利用开始的IMU帧的平均值初始化状态量x]
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N)
{
  //MeasureGroup is a struct representing all data being processed in the current process, containing IMU queue, one frame of lidar point cloud, and lidar start and end times[MeasureGroup这个struct表示当前过程中正在处理的所有数据，包含IMU队列和一帧lidar的点云 以及lidar的起始和结束时间]
  //Initialize gravity, gyroscope bias, acceleration and gyroscope covariance. Normalize acceleration measurements to unit gravity[初始化重力、陀螺仪偏差、acc和陀螺仪协方差  将加速度测量值归一化为单位重力   **/
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_) //If it is the first IMU frame[如果为第一帧IMU]
  {
    Reset();    //Reset IMU parameters[重置IMU参数]
    N = 1;      //Set iteration count to 1[将迭代次数置1]
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;    //Acceleration at initial IMU time[IMU初始时刻的加速度]
    const auto &gyr_acc = meas.imu.front()->angular_velocity;       //Angular velocity at initial IMU time[IMU初始时刻的角速度]
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;              //Use first frame acceleration value as initialization mean[第一帧加速度值作为初始化均值]
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;              //Use first frame angular velocity value as initialization mean[第一帧角速度值作为初始化均值]
    first_lidar_time = meas.lidar_beg_time;                   //Use current IMU frame's corresponding lidar start time as initial time[将当前IMU帧对应的lidar起始时间 作为初始时间]
  }

  for (const auto &imu : meas.imu)    //Calculate mean and variance based on all IMU data[根据所有IMU数据，计算平均值和方差]
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc  += (cur_acc - mean_acc) / N;    //Update mean based on difference between current frame and mean[根据当前帧和均值差作为均值的更新]
    mean_gyr  += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc)  / N;
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr)  / N / N * (N-1);

    N ++;
  }
  
  state_ikfom init_state = kf_state.get_x();        //Get state x_ from esekfom.hpp[在esekfom.hpp获得x_的状态]
  init_state.grav = - mean_acc / mean_acc.norm() * G_m_s2;    //Get unit direction vector of average measurement * preset gravity value[得平均测量的单位方向向量 * 重力加速度预设值]
  
  init_state.bg  = mean_gyr;      //Use angular velocity measurement as gyroscope bias[角速度测量作为陀螺仪偏差]
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;      //Pass in lidar and imu extrinsic parameters[将lidar和imu外参传入]
  init_state.offset_R_L_I = Sophus::SO3(Lidar_R_wrt_IMU);
  kf_state.change_x(init_state);      //Pass initialized state to x_ in esekfom.hpp[将初始化后的状态传入esekfom.hpp中的x_]

  Matrix<double, 24, 24> init_P = MatrixXd::Identity(24,24);      //Get covariance matrix P_ from esekfom.hpp[在esekfom.hpp获得P_的协方差矩阵]
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = init_P(23,23) = 0.00001; 
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();

  // std::cout << "IMU init new -- init_state  " << init_state.pos  <<" " << init_state.bg <<" " << init_state.ba <<" " << init_state.grav << std::endl;
}

//Backward propagation[反向传播]
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI &pcl_out)
{
  //Add the last IMU of the previous frame to the beginning of the current frame IMU [将上一帧最后尾部的imu添加到当前帧头部的imu]
  auto v_imu = meas.imu;         //Extract IMU queue of current frame[取出当前帧的IMU队列]
  v_imu.push_front(last_imu_);   //Add last IMU of previous frame to beginning of current frame[将上一帧最后尾部的imu添加到当前帧头部的imu]
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();    //Get the time of IMU at end of current frame[拿到当前帧尾部的imu的时间]
  const double &pcl_beg_time = meas.lidar_beg_time;      // Point cloud start and end timestamps[点云开始和结束的时间戳]
  const double &pcl_end_time = meas.lidar_end_time;
  
  // Re-sort point cloud based on timestamp of each point[根据点云中每个点的时间戳对点云进行重排序]
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);  //Here curvature stores the timestamp (in preprocess.cpp)[这里curvature中存放了时间戳（在preprocess.cpp中）]


  state_ikfom imu_state = kf_state.get_x();  // Get posterior state from last KF estimation as initial state for this IMU prediction[获取上一次KF估计的后验状态作为本次IMU预测的初始状态]
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));
  //Add initial state to IMUpose, containing time interval, last frame acceleration, last frame angular velocity, last frame velocity, last frame position, last frame rotation matrix[将初始状态加入IMUpose中,包含有时间间隔，上一帧加速度，上一帧角速度，上一帧速度，上一帧位置，上一帧旋转矩阵]

  // Forward propagation [前向传播]
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu; // angvel_avr is average angular velocity, acc_avr is average acceleration, acc_imu is IMU acceleration, vel_imu is IMU velocity, pos_imu is IMU position[angvel_avr为平均角速度，acc_avr为平均加速度，acc_imu为imu加速度，vel_imu为imu速度，pos_imu为imu位置]
  M3D R_imu;    //IMU rotation matrix used for motion distortion elimination[IMU旋转矩阵 消除运动失真的时候用]

  double dt = 0;

  input_ikfom in;
  // Traverse all IMU measurements for this estimation and integrate using discrete midpoint method for forward propagation[遍历本次估计的所有IMU测量并且进行积分，离散中值法 前向传播]
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);        //Get current frame IMU data[拿到当前帧的imu数据]
    auto &&tail = *(it_imu + 1);    //Get next frame IMU data[拿到下一帧的imu数据]
    //Determine time order: whether next frame timestamp is less than last frame end timestamp, if not, continue directly[判断时间先后顺序：下一帧时间戳是否小于上一帧结束时间戳 不符合直接continue]
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),      // Midpoint integration[中值积分]
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    acc_avr  = acc_avr * G_m_s2 / mean_acc.norm(); //Adjust acceleration using gravity value (divide by initialized IMU magnitude * 9.8)[通过重力数值对加速度进行调整(除上初始化的IMU大小*9.8)]

    //If IMU start time is earlier than last radar end time (since the last IMU of the previous frame is inserted at the beginning of this frame, this will occur once)[如果IMU开始时刻早于上次雷达最晚时刻(因为将上次最后一个IMU插入到此次开头了，所以会出现一次这种情况)]
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_; //Start propagation from end of last radar time, calculate time difference with end of this IMU[从上次雷达时刻末尾开始传播 计算与此次IMU结尾之间的时间差]
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();     //Time interval between two IMU times[两个IMU时刻之间的时间间隔]
    }
    
    in.acc = acc_avr;     // Midpoint of two IMU frames as input for forward propagation[两帧IMU的中值作为输入in  用于前向传播]
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;         // Configure covariance matrix[配置协方差矩阵]
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;

    kf_state.predict(dt, Q, in);    // IMU forward propagation, time interval for each propagation is dt[IMU前向传播，每次传播的时间间隔为dt]

    imu_state = kf_state.get_x();   //Update IMU state to integrated state[更新IMU状态为积分后的状态]
    //Update last frame angular velocity = next frame angular velocity - bias[更新上一帧角速度 = 后一帧角速度-bias]
    angvel_last = V3D(tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z) - imu_state.bg;
    //Update acceleration in world coordinate system = R*(acceleration - bias) - g[更新上一帧世界坐标系下的加速度 = R*(加速度-bias) - g]
    acc_s_last  = V3D(tail->linear_acceleration.x, tail->linear_acceleration.y, tail->linear_acceleration.z) * G_m_s2 / mean_acc.norm();   

    // std::cout << "acc_s_last: " << acc_s_last.transpose() << std::endl;
    // std::cout << "imu_state.ba: " << imu_state.ba.transpose() << std::endl;
    // std::cout << "imu_state.grav: " << imu_state.grav.transpose() << std::endl;
    acc_s_last = imu_state.rot * (acc_s_last - imu_state.ba) + imu_state.grav;
    // std::cout << "--acc_s_last: " << acc_s_last.transpose() << std::endl<< std::endl;

    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;    //Time interval from next IMU time to current radar start[后一个IMU时刻距离此次雷达开始的时间间隔]
    IMUpose.push_back( set_pose6d( offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix() ) );
  }

  // Also add the last IMU measurement[把最后一帧IMU测量也补上]
  dt = abs(pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);
  imu_state = kf_state.get_x();   
  last_imu_ = meas.imu.back();              //Save last IMU measurement for use in next frame[保存最后一个IMU测量，以便于下一帧使用]
  last_lidar_end_time_ = pcl_end_time;      //Save end time of last radar measurement in this frame for use in next frame[保存这一帧最后一个雷达测量的结束时间，以便于下一帧使用]

  // Remove distortion from each lidar point (backward propagation) [消除每个激光雷达点的失真（反向传播）]
  if (pcl_out.points.begin() == pcl_out.points.end()) return;
  auto it_pcl = pcl_out.points.end() - 1;

  //Traverse each IMU frame[遍历每个IMU帧]
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);   
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);    
    pos_imu<<VEC_FROM_ARRAY(head->pos);    
    acc_imu<<VEC_FROM_ARRAY(tail->acc);    
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr); 


    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;    //Time interval from point to IMU start time[点到IMU开始时刻的时间间隔]

      /*    P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei)    */

      M3D R_i(R_imu * Sophus::SO3::exp(angvel_avr * dt).matrix() );   //Rotation at time of point it_pcl: previous frame IMU rotation matrix * exp(next frame angular velocity * dt)[点it_pcl所在时刻的旋转：前一帧的IMU旋转矩阵 * exp(后一帧角速度*dt)]
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);   //Position at time of point (in radar coordinate system)[点所在时刻的位置(雷达坐标系下)]
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);   //From point's world position - radar end world position[从点所在的世界位置-雷达末尾世界位置]
      V3D P_compensate = imu_state.offset_R_L_I.matrix().transpose() * (imu_state.rot.matrix().transpose() * (R_i * (imu_state.offset_R_L_I.matrix() * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}


double T1,T2;
void ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI::Ptr &cur_pcl_un_)
{
  // T1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)   
  {
    // The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);  

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();

    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
    }

    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_); 

  // T2 = omp_get_wtime();
  // cout<<"[ IMU Process ]: Time: "<<T2 - T1<<endl;
}
