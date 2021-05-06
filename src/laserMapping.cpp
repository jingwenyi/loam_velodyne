// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

#include <chrono>
#include <math.h>

#include <loam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

struct FreqReport {
  std::string name;
  std::chrono::system_clock::time_point last_time;
  bool firstTime;
  FreqReport(const std::string &n) : name(n), firstTime(true) {}
  void report() {
    if (firstTime) {
      firstTime = false;
      last_time = std::chrono::system_clock::now();
      return;
    }
    auto cur_time = std::chrono::system_clock::now();
    ROS_INFO("time interval of %s = %f seconds\n", name.c_str(),
             std::chrono::duration_cast<
                 std::chrono::duration<float, std::ratio<1, 1>>>(cur_time -
                                                                 last_time)
                 .count());
    last_time = cur_time;
  }
};

const float scanPeriod = 0.1;

#ifndef VELODYNE_HDL64E
//控制接收到的点云数据，每隔几帧处理一次
const int stackFrameNum = 1;
//控制处理得到的点云map, 每隔几次publish 给rviz 显示
const int mapFrameNum = 5;
const int maxIterNum = 10;
#else
const int stackFrameNum = 1;
const int mapFrameNum = 5;
const int maxIterNum = 20;
#endif


double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;

bool newLaserCloudCornerLast = false;
bool newLaserCloudSurfLast = false;
bool newLaserCloudFullRes = false;
bool newLaserOdometry = false;

//这里是地图中心(center) 在地图中的位置的初始化值
//作者把地图原点初始化值放在整个可建图范围的中心
int laserCloudCenWidth = 10;
int laserCloudCenHeight = 5;
int laserCloudCenDepth = 10;



//地图各个方向的cube 个数，真实的地图缩小了50倍，
//从而可以计算出扎个地图在width 方向 能够表示的距离是50*21 = 1050 m
//在height 的方向上能够表示的距离是11 * 50 = 550m
//在depth 的方向上能够表示的距离是21 * 50 = 1050m
//子cube  沿宽方向的分割个数
const int laserCloudWidth = 21;
//子cube 沿高方向的分割个数
const int laserCloudHeight = 11;
//子 cube 沿深度方向分割个数
const int laserCloudDepth = 21;


//地图的大小, 中体积为1050 * 550 * 1050 = 606375000 m*m*m
//子cube 总数4851
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;

//lidar 视域范围内(FOV) 的点云集索引
int laserCloudValidInd[125];
//lidar 周围点云集索引
int laserCloudSurroundInd[125];


//最新接收到的特征点(边缘 + 平面)

pcl::PointCloud<PointType>::Ptr
    laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    laserCloudSurfLast(new pcl::PointCloud<PointType>());


//存放当前收到的特征点(下采样之后的)
pcl::PointCloud<PointType>::Ptr
    laserCloudCornerStack(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    laserCloudSurfStack(new pcl::PointCloud<PointType>());

//存放当前收到的特征点(下采样之前)
pcl::PointCloud<PointType>::Ptr
    laserCloudCornerStack2(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    laserCloudSurfStack2(new pcl::PointCloud<PointType>());


//原始点云坐标和权重
pcl::PointCloud<PointType>::Ptr laserCloudOri(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr coeffSel(new pcl::PointCloud<PointType>());


//匹配使用的特征点(下采样之后)
pcl::PointCloud<PointType>::Ptr
    laserCloudSurround(new pcl::PointCloud<PointType>());
//匹配使用的特征点(下采样之前)
pcl::PointCloud<PointType>::Ptr
    laserCloudSurround2(new pcl::PointCloud<PointType>());



//map 中提取的匹配使用的特征点
pcl::PointCloud<PointType>::Ptr
    laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    laserCloudSurfFromMap(new pcl::PointCloud<PointType>());


//点云全部点
pcl::PointCloud<PointType>::Ptr
    laserCloudFullRes(new pcl::PointCloud<PointType>());


//array 都是以50 m 为单位的立方体地图，运行过程中会一直保存
//(有需要的话可以考虑优化，只保存邻近的)

//存放特征点的立方体cube
//这里是把一个三维数组保存在一个一维数组中 [21][11][21]
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

//中间变量，存放下采样的特征点

pcl::PointCloud<PointType>::Ptr laserCloudCornerArray2[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray2[laserCloudNum];



//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr
    kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr
    kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

// rotation {x, y, z} or {pitch?, yaw?, roll?}, translation {x, y, z}
//当前帧结束点到第一帧开始点的状态转移矩阵，也就是里程计



//odometry 计算得到的到世界坐标系下的转移矩阵T
float transformSum[6] = {0};

//转移增量，只使用了后三个平移增量
float transformIncre[6] = {0};


//以起始位置为原点的世界坐标系下的转换矩阵(猜测与调整的对象)
float transformTobeMapped[6] = {0};

//存放mapping 之前的Odometry 计算的世界坐标系的转移矩阵(低频量，不一定于transformSum 一样)
float transformBefMapped[6] = {0};

//存放mapping 之后的经过mapping 微调之后的转换矩阵
float transformAftMapped[6] = {0};

int imuPointerFront = 0;
int imuPointerLast = -1;
const int imuQueLength = 200;

double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};

//基于匀速模型，根据上次微调的结果和odometry 这次与上次计算的结果
//猜测一个新的世界坐标系的转换矩阵transformTobeMapped


void transformAssociateToMap() {
	//把上一次建图时使用的建图帧结束点云相对于 
	//第一帧开始点云的里程计状态矩阵
	//旋转到当前帧结束点云的机体坐标系中

	//  先平移，再 绕y 轴 (-yaw) 旋转
  float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) -
             sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
  float y1 = transformBefMapped[4] - transformSum[4];
  float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) +
             cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
	//绕x 轴(-pitch) 旋转
  float x2 = x1;
  float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
  float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

	//绕z 轴(-roll) 旋转
  transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
  transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
  transformIncre[5] = z2;

  float sbcx = sin(transformSum[0]);
  float cbcx = cos(transformSum[0]);
  float sbcy = sin(transformSum[1]);
  float cbcy = cos(transformSum[1]);
  float sbcz = sin(transformSum[2]);
  float cbcz = cos(transformSum[2]);

  float sblx = sin(transformBefMapped[0]);
  float cblx = cos(transformBefMapped[0]);
  float sbly = sin(transformBefMapped[1]);
  float cbly = cos(transformBefMapped[1]);
  float sblz = sin(transformBefMapped[2]);
  float cblz = cos(transformBefMapped[2]);

  float salx = sin(transformAftMapped[0]);
  float calx = cos(transformAftMapped[0]);
  float saly = sin(transformAftMapped[1]);
  float caly = cos(transformAftMapped[1]);
  float salz = sin(transformAftMapped[2]);
  float calz = cos(transformAftMapped[2]);

	//通过上一次建图时使用的 旋转矩阵(即里程计)， 
	//和当前帧的旋转矩阵，求出当前帧
	//对应的地图旋转矩阵，从而求出位姿转移矩阵
  //R_transformTobeMapped(n) = R_transfromAftMapped(n-1)^T * R_transfromSum(n)

  float srx = -sbcx * (salx * sblx + calx * cblx * salz * sblz +
                       calx * calz * cblx * cblz) -
              cbcx * sbcy * (calx * calz * (cbly * sblz - cblz * sblx * sbly) -
                             calx * salz * (cbly * cblz + sblx * sbly * sblz) +
                             cblx * salx * sbly) -
              cbcx * cbcy * (calx * salz * (cblz * sbly - cbly * sblx * sblz) -
                             calx * calz * (sbly * sblz + cbly * cblz * sblx) +
                             cblx * cbly * salx);
  transformTobeMapped[0] = -asin(srx);

  float srycrx = sbcx * (cblx * cblz * (caly * salz - calz * salx * saly) -
                         cblx * sblz * (caly * calz + salx * saly * salz) +
                         calx * saly * sblx) -
                 cbcx * cbcy * ((caly * calz + salx * saly * salz) *
                                    (cblz * sbly - cbly * sblx * sblz) +
                                (caly * salz - calz * salx * saly) *
                                    (sbly * sblz + cbly * cblz * sblx) -
                                calx * cblx * cbly * saly) +
                 cbcx * sbcy * ((caly * calz + salx * saly * salz) *
                                    (cbly * cblz + sblx * sbly * sblz) +
                                (caly * salz - calz * salx * saly) *
                                    (cbly * sblz - cblz * sblx * sbly) +
                                calx * cblx * saly * sbly);
  float crycrx = sbcx * (cblx * sblz * (calz * saly - caly * salx * salz) -
                         cblx * cblz * (saly * salz + caly * calz * salx) +
                         calx * caly * sblx) +
                 cbcx * cbcy * ((saly * salz + caly * calz * salx) *
                                    (sbly * sblz + cbly * cblz * sblx) +
                                (calz * saly - caly * salx * salz) *
                                    (cblz * sbly - cbly * sblx * sblz) +
                                calx * caly * cblx * cbly) -
                 cbcx * sbcy * ((saly * salz + caly * calz * salx) *
                                    (cbly * sblz - cblz * sblx * sbly) +
                                (calz * saly - caly * salx * salz) *
                                    (cbly * cblz + sblx * sbly * sblz) -
                                calx * caly * cblx * sbly);
  transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]),
                                 crycrx / cos(transformTobeMapped[0]));

  float srzcrx = (cbcz * sbcy - cbcy * sbcx * sbcz) *
                     (calx * salz * (cblz * sbly - cbly * sblx * sblz) -
                      calx * calz * (sbly * sblz + cbly * cblz * sblx) +
                      cblx * cbly * salx) -
                 (cbcy * cbcz + sbcx * sbcy * sbcz) *
                     (calx * calz * (cbly * sblz - cblz * sblx * sbly) -
                      calx * salz * (cbly * cblz + sblx * sbly * sblz) +
                      cblx * salx * sbly) +
                 cbcx * sbcz * (salx * sblx + calx * cblx * salz * sblz +
                                calx * calz * cblx * cblz);
  float crzcrx = (cbcy * sbcz - cbcz * sbcx * sbcy) *
                     (calx * calz * (cbly * sblz - cblz * sblx * sbly) -
                      calx * salz * (cbly * cblz + sblx * sbly * sblz) +
                      cblx * salx * sbly) -
                 (sbcy * sbcz + cbcy * cbcz * sbcx) *
                     (calx * salz * (cblz * sbly - cbly * sblx * sblz) -
                      calx * calz * (sbly * sblz + cbly * cblz * sblx) +
                      cblx * cbly * salx) +
                 cbcx * cbcz * (salx * sblx + calx * cblx * salz * sblz +
                                calx * calz * cblx * cblz);
  transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]),
                                 crzcrx / cos(transformTobeMapped[0]));

  //把求出的当前帧对应的地图姿态矩阵，从当前帧的机体坐标转到世界坐标

	//绕z 轴旋转(roll)
  x1 = cos(transformTobeMapped[2]) * transformIncre[3] -
       sin(transformTobeMapped[2]) * transformIncre[4];
  y1 = sin(transformTobeMapped[2]) * transformIncre[3] +
       cos(transformTobeMapped[2]) * transformIncre[4];
  z1 = transformIncre[5];

	//绕x 轴旋转(pitch)
  x2 = x1;
  y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
  z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

	//绕 y 轴旋转(yaw), 再平移
  transformTobeMapped[3] =
      transformAftMapped[3] -
      (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
  transformTobeMapped[4] = transformAftMapped[4] - y2;
  transformTobeMapped[5] =
      transformAftMapped[5] -
      (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
}

//记录odometry 发送的转换矩阵与mapping之后的转换矩阵
void transformUpdate() {
  if (imuPointerLast >= 0) {
    float imuRollLast = 0, imuPitchLast = 0;
	//查找点云时间戳小于imu 时间戳的imu 位置
    while (imuPointerFront != imuPointerLast) {
      if (timeLaserOdometry + scanPeriod < imuTime[imuPointerFront]) {
        break;
      }
      imuPointerFront = (imuPointerFront + 1) % imuQueLength;
    }

    if (timeLaserOdometry + scanPeriod > imuTime[imuPointerFront]) {
		//未找到，此时imuPointerFront == imuPointerLast
      imuRollLast = imuRoll[imuPointerFront];
      imuPitchLast = imuPitch[imuPointerFront];
    } else {
      int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
      float ratioFront =
          (timeLaserOdometry + scanPeriod - imuTime[imuPointerBack]) /
          (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
      float ratioBack =
          (imuTime[imuPointerFront] - timeLaserOdometry - scanPeriod) /
          (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
		//按时间比例求翻滚角和俯仰角
      imuRollLast = imuRoll[imuPointerFront] * ratioFront +
                    imuRoll[imuPointerBack] * ratioBack;
      imuPitchLast = imuPitch[imuPointerFront] * ratioFront +
                     imuPitch[imuPointerBack] * ratioBack;
    }

	//对imu 数据进行低通滤波
    transformTobeMapped[0] =
        0.998 * transformTobeMapped[0] + 0.002 * imuPitchLast;
    transformTobeMapped[2] =
        0.998 * transformTobeMapped[2] + 0.002 * imuRollLast;
  }

	//记录优化之前与优化之后的转移矩阵
  for (int i = 0; i < 6; i++) {
    transformBefMapped[i] = transformSum[i];
    transformAftMapped[i] = transformTobeMapped[i];
  }
}


//根据调整计算后的转移矩阵，将点注册到全局世界坐标系下
void pointAssociateToMap(PointType const *const pi, PointType *const po) {
	//绕z 轴(roll) 旋转(transformToMapped[2])
  float x1 =
      cos(transformTobeMapped[2]) * pi->x - sin(transformTobeMapped[2]) * pi->y;
  float y1 =
      sin(transformTobeMapped[2]) * pi->x + cos(transformTobeMapped[2]) * pi->y;
  float z1 = pi->z;

	//绕x 轴(pitch) 旋转(transformToMapped[0])
  float x2 = x1;
  float y2 =
      cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
  float z2 =
      sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

	//绕y 轴(yaw) 旋转(transformToMapped[1])，再平移
  po->x = cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2 +
          transformTobeMapped[3];
  po->y = y2 + transformTobeMapped[4];
  po->z = -sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2 +
          transformTobeMapped[5];
  po->intensity = pi->intensity;
}

//点转移到局部坐标系下
void pointAssociateTobeMapped(PointType const *const pi, PointType *const po) {
//平移后绕y 轴旋转(-transformTobeMapped[1])
  float x1 = cos(transformTobeMapped[1]) * (pi->x - transformTobeMapped[3]) -
             sin(transformTobeMapped[1]) * (pi->z - transformTobeMapped[5]);
  float y1 = pi->y - transformTobeMapped[4];
  float z1 = sin(transformTobeMapped[1]) * (pi->x - transformTobeMapped[3]) +
             cos(transformTobeMapped[1]) * (pi->z - transformTobeMapped[5]);
//绕x 轴旋转(-transformTobeMapped[0])
  float x2 = x1;
  float y2 =
      cos(transformTobeMapped[0]) * y1 + sin(transformTobeMapped[0]) * z1;
  float z2 =
      -sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;
//绕z 轴旋转(-transformTobeMapp[2])
  po->x = cos(transformTobeMapped[2]) * x2 + sin(transformTobeMapped[2]) * y2;
  po->y = -sin(transformTobeMapped[2]) * x2 + cos(transformTobeMapped[2]) * y2;
  po->z = z2;
  po->intensity = pi->intensity;
}

//接收边沿点
void laserCloudCornerLastHandler(
    const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2) {
  timeLaserCloudCornerLast = laserCloudCornerLast2->header.stamp.toSec();

  laserCloudCornerLast->clear();
  pcl::fromROSMsg(*laserCloudCornerLast2, *laserCloudCornerLast);

  newLaserCloudCornerLast = true;
}


//接收平面点
void laserCloudSurfLastHandler(
    const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2) {
  timeLaserCloudSurfLast = laserCloudSurfLast2->header.stamp.toSec();

  laserCloudSurfLast->clear();
  pcl::fromROSMsg(*laserCloudSurfLast2, *laserCloudSurfLast);

  newLaserCloudSurfLast = true;
}

//接收点云全部点
void laserCloudFullResHandler(
    const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2) {
  timeLaserCloudFullRes = laserCloudFullRes2->header.stamp.toSec();

  laserCloudFullRes->clear();
  pcl::fromROSMsg(*laserCloudFullRes2, *laserCloudFullRes);

  newLaserCloudFullRes = true;
}

//接收旋转平移信息
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry) {
  timeLaserOdometry = laserOdometry->header.stamp.toSec();

  double roll, pitch, yaw;
  //四元数转换为欧拉角
  geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
  tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w))
      .getRPY(roll, pitch, yaw);

  transformSum[0] = -pitch;
  transformSum[1] = -yaw;
  transformSum[2] = roll;

  transformSum[3] = laserOdometry->pose.pose.position.x;
  transformSum[4] = laserOdometry->pose.pose.position.y;
  transformSum[5] = laserOdometry->pose.pose.position.z;

  newLaserOdometry = true;
}

//接收IMU 信息，只使用了翻滚角和俯仰角
void imuHandler(const sensor_msgs::Imu::ConstPtr &imuIn) {
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(imuIn->orientation, orientation);
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  imuPointerLast = (imuPointerLast + 1) % imuQueLength;

  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = roll;
  imuPitch[imuPointerLast] = pitch;
}

FreqReport mappingCloudFreq("mappingCloud");
FreqReport mappingOdometryFreq("mappingOdometry");

int main(int argc, char **argv) {
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;

  ros::Subscriber subLaserCloudCornerLast =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2,
                                             laserCloudCornerLastHandler);

  ros::Subscriber subLaserCloudSurfLast =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2,
                                             laserCloudSurfLastHandler);

  ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>(
      "/laser_odom_to_init", 5, laserOdometryHandler);

  ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>(
      "/velodyne_cloud_3", 2, laserCloudFullResHandler);

  ros::Subscriber subImu =
      nh.subscribe<sensor_msgs::Imu>("/imu/data", 50, imuHandler);

  ros::Publisher pubLaserCloudSurround =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 1);

  ros::Publisher pubLaserCloudFullRes =
      nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 2);

  ros::Publisher pubOdomAftMapped =
      nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 5);
  nav_msgs::Odometry odomAftMapped;
  odomAftMapped.header.frame_id = "/camera_init";
  odomAftMapped.child_frame_id = "/aft_mapped";

  tf::TransformBroadcaster tfBroadcaster;
  tf::StampedTransform aftMappedTrans;
  aftMappedTrans.frame_id_ = "/camera_init";
  aftMappedTrans.child_frame_id_ = "/aft_mapped";

  std::vector<int> pointSearchInd;
  std::vector<float> pointSearchSqDis;

  PointType pointOri, pointSel, pointProj, coeff;

  cv::Mat matA0(5, 3, CV_32F, cv::Scalar::all(0));
  cv::Mat matB0(5, 1, CV_32F, cv::Scalar::all(-1));
  cv::Mat matX0(3, 1, CV_32F, cv::Scalar::all(0));

  cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
  cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
  cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

  bool isDegenerate = false;
  cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

	//创建体素栅格滤波器
  pcl::VoxelGrid<PointType> downSizeFilterCorner;
	//设置体素大小
  downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);

  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);

  pcl::VoxelGrid<PointType> downSizeFilterMap;
  downSizeFilterMap.setLeafSize(0.6, 0.6, 0.6);

	//指针初始化
  for (int i = 0; i < laserCloudNum; i++) {
    laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
    laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
    laserCloudCornerArray2[i].reset(new pcl::PointCloud<PointType>());
    laserCloudSurfArray2[i].reset(new pcl::PointCloud<PointType>());
  }

  int frameCount = stackFrameNum - 1; //0
  int mapFrameCount = mapFrameNum - 1; // 4
  ros::Rate rate(100);
  bool status = ros::ok();
  while (status) {
    ros::spinOnce();

    if (newLaserCloudCornerLast && newLaserCloudSurfLast &&
        newLaserCloudFullRes && newLaserOdometry &&
        fabs(timeLaserCloudCornerLast - timeLaserOdometry) < 0.005 &&
        fabs(timeLaserCloudSurfLast - timeLaserOdometry) < 0.005 &&
        fabs(timeLaserCloudFullRes - timeLaserOdometry) < 0.005) {
      newLaserCloudCornerLast = false;
      newLaserCloudSurfLast = false;
      newLaserCloudFullRes = false;
      newLaserOdometry = false;

      frameCount++;
      if (frameCount >= stackFrameNum) {
	  	//通过上一帧的地图位姿 矩阵和当前帧的里程计求当前帧的地图位姿矩阵
        transformAssociateToMap();


		//把当前边沿点旋转到最新地图坐标系中
        int laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        for (int i = 0; i < laserCloudCornerLastNum; i++) {
			//把点旋转到新的世界坐标系中
          pointAssociateToMap(&laserCloudCornerLast->points[i], &pointSel);
		  //把旋转到最新地图坐标系中的边沿特征点，保存到laserCloudCornerStack2 中
          laserCloudCornerStack2->push_back(pointSel);
        }

		//把当前平面点旋转到最新地图坐标系中
        int laserCloudSurfLastNum = laserCloudSurfLast->points.size();
        for (int i = 0; i < laserCloudSurfLastNum; i++) {
			//把点关联到最新的地图坐标系中
          pointAssociateToMap(&laserCloudSurfLast->points[i], &pointSel);
			//把关联到世界坐标系的平面特征点，保存到laserCloudSurfStack2 中
          laserCloudSurfStack2->push_back(pointSel);
        }
      }

      if (frameCount >= stackFrameNum) {
        frameCount = 0;

        PointType pointOnYAxis;
        pointOnYAxis.x = 0.0;
        pointOnYAxis.y = 10.0;
        pointOnYAxis.z = 0.0;
		//获取y 方向上10 m 高位置的点在世界坐标系下的坐标
        pointAssociateToMap(&pointOnYAxis, &pointOnYAxis);

		//立方体中点在世界坐标系下的(原点) 位置
		//过半取一(以50 m 进行四舍五入的效果)，由于数组下标
		//为正数，而地图可能建立在原点前后，因此
		//每一维偏移一个laserCloudCenWidth (该值会动态调整，以
		//使的数组利用最大化，初始值为该数组长度1/2) 的量



		//通过transformAssociateToMap 猜测的新的世界坐标的旋转矩阵中的平移量
		//求出该点云帧的中心点在地图中的那个立方体中
        int centerCubeI =
            int((transformTobeMapped[3] + 25.0) / 50.0) + laserCloudCenWidth;
        int centerCubeJ =
            int((transformTobeMapped[4] + 25.0) / 50.0) + laserCloudCenHeight;
        int centerCubeK =
            int((transformTobeMapped[5] + 25.0) / 50.0) + laserCloudCenDepth;

		//由于计算机求余是向零取整，为了不使(-50.0,50.0) 求余后
		//都向零偏移，当被求余数为负数时求余结果统一左移
		//一个单位，也即减一
        if (transformTobeMapped[3] + 25.0 < 0)
          centerCubeI--;
        if (transformTobeMapped[4] + 25.0 < 0)
          centerCubeJ--;
        if (transformTobeMapped[5] + 25.0 < 0)
          centerCubeK--;



		//调整整个地图在该三维空间(其实是一个一维数组) 中的位置，防止地图当前扫描到的数据
		//越界，按照该思路，可以认为如果一直向一个方面建图，超过那个方向能够保持的
		//距离后，就会丢掉之前的地图数据


		//由于激光雷达扫描范围是150m 内，故地图原点中心离边界，至少有3 个立方体
		//调整之后取值范围: 3< centerCubeI < 18, 3 < centerCubeJ < 8,
		// 3<centerCubeK < 18

		//如果处于下边界，表明地图向负方向延伸的可能性比较大,
		//则循环移位，将数组中心点向上边界调整一个单位

        // shift block of point cloud

        while (centerCubeI < 3) {
          for (int j = 0; j < laserCloudHeight; j++) {
            for (int k = 0; k < laserCloudDepth; k++) {
				//实现一次循环移位的效果
              int i = laserCloudWidth - 1;
				//指针赋值，保存最后一个指针位置
				//这里可以理解成一个三维数组保存在一个一维的数组中
				//所以最后一个立方体是i + 
              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                  laserCloudCornerArray[i + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight * k];
              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                  laserCloudSurfArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
				//循环移位，I 纬度上依次后移
			  for (; i >= 1; i--) {
                laserCloudCornerArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k] =
                    laserCloudCornerArray[i - 1 + laserCloudWidth * j +
                                          laserCloudWidth * laserCloudHeight *
                                              k];
                laserCloudSurfArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                    laserCloudSurfArray[i - 1 + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight * k];
              }

			  //将开始点赋值为最后一个点
              laserCloudCornerArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCubeCornerPointer;
              laserCloudSurfArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCubeSurfPointer;
              laserCloudCubeCornerPointer->clear();
              laserCloudCubeSurfPointer->clear();
            }
          }

          centerCubeI++;
          laserCloudCenWidth++;
        }


		//如果处于上边界，表明地图向正方向延伸的可能性比较大，
		//则循环移位，将数组中心点向下边界调整一个单位
        while (centerCubeI >= laserCloudWidth - 3) {
          for (int j = 0; j < laserCloudHeight; j++) {
            for (int k = 0; k < laserCloudDepth; k++) {
              int i = 0;
              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                  laserCloudCornerArray[i + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight * k];
              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                  laserCloudSurfArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
				//I 纬度上依次前移
			  for (; i < laserCloudWidth - 1; i++) {
                laserCloudCornerArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k] =
                    laserCloudCornerArray[i + 1 + laserCloudWidth * j +
                                          laserCloudWidth * laserCloudHeight *
                                              k];
                laserCloudSurfArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                    laserCloudSurfArray[i + 1 + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight * k];
              }
              laserCloudCornerArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCubeCornerPointer;
              laserCloudSurfArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCubeSurfPointer;
              laserCloudCubeCornerPointer->clear();
              laserCloudCubeSurfPointer->clear();
            }
          }

          centerCubeI--;
          laserCloudCenWidth--;
        }

        while (centerCubeJ < 3) {
          for (int i = 0; i < laserCloudWidth; i++) {
            for (int k = 0; k < laserCloudDepth; k++) {
              int j = laserCloudHeight - 1;
              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                  laserCloudCornerArray[i + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight * k];
              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                  laserCloudSurfArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
				//J 纬度上，依次后移
			  for (; j >= 1; j--) {
                laserCloudCornerArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k] =
                    laserCloudCornerArray[i + laserCloudWidth * (j - 1) +
                                          laserCloudWidth * laserCloudHeight *
                                              k];
                laserCloudSurfArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                    laserCloudSurfArray[i + laserCloudWidth * (j - 1) +
                                        laserCloudWidth * laserCloudHeight * k];
              }
              laserCloudCornerArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCubeCornerPointer;
              laserCloudSurfArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCubeSurfPointer;
              laserCloudCubeCornerPointer->clear();
              laserCloudCubeSurfPointer->clear();
            }
          }

          centerCubeJ++;
          laserCloudCenHeight++;
        }

        while (centerCubeJ >= laserCloudHeight - 3) {
          for (int i = 0; i < laserCloudWidth; i++) {
            for (int k = 0; k < laserCloudDepth; k++) {
              int j = 0;
              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                  laserCloudCornerArray[i + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight * k];
              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                  laserCloudSurfArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
				//J 维度上依次前移
			  for (; j < laserCloudHeight - 1; j++) {
                laserCloudCornerArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k] =
                    laserCloudCornerArray[i + laserCloudWidth * (j + 1) +
                                          laserCloudWidth * laserCloudHeight *
                                              k];
                laserCloudSurfArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                    laserCloudSurfArray[i + laserCloudWidth * (j + 1) +
                                        laserCloudWidth * laserCloudHeight * k];
              }
              laserCloudCornerArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCubeCornerPointer;
              laserCloudSurfArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCubeSurfPointer;
              laserCloudCubeCornerPointer->clear();
              laserCloudCubeSurfPointer->clear();
            }
          }

          centerCubeJ--;
          laserCloudCenHeight--;
        }

        while (centerCubeK < 3) {
          for (int i = 0; i < laserCloudWidth; i++) {
            for (int j = 0; j < laserCloudHeight; j++) {
              int k = laserCloudDepth - 1;
              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                  laserCloudCornerArray[i + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight * k];
              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                  laserCloudSurfArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
				//K 维度上依次后移
			  for (; k >= 1; k--) {
                laserCloudCornerArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k] =
                    laserCloudCornerArray[i + laserCloudWidth * j +
                                          laserCloudWidth * laserCloudHeight *
                                              (k - 1)];
                laserCloudSurfArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                    laserCloudSurfArray[i + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight *
                                            (k - 1)];
              }
              laserCloudCornerArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCubeCornerPointer;
              laserCloudSurfArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCubeSurfPointer;
              laserCloudCubeCornerPointer->clear();
              laserCloudCubeSurfPointer->clear();
            }
          }

          centerCubeK++;
          laserCloudCenDepth++;
        }

        while (centerCubeK >= laserCloudDepth - 3) {
          for (int i = 0; i < laserCloudWidth; i++) {
            for (int j = 0; j < laserCloudHeight; j++) {
              int k = 0;
              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                  laserCloudCornerArray[i + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight * k];
              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                  laserCloudSurfArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
				//k维度上依次前移
			  for (; k < laserCloudDepth - 1; k++) {
                laserCloudCornerArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k] =
                    laserCloudCornerArray[i + laserCloudWidth * j +
                                          laserCloudWidth * laserCloudHeight *
                                              (k + 1)];
                laserCloudSurfArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                    laserCloudSurfArray[i + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight *
                                            (k + 1)];
              }
              laserCloudCornerArray[i + laserCloudWidth * j +
                                    laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCubeCornerPointer;
              laserCloudSurfArray[i + laserCloudWidth * j +
                                  laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCubeSurfPointer;
              laserCloudCubeCornerPointer->clear();
              laserCloudCubeSurfPointer->clear();
            }
          }

          centerCubeK--;
          laserCloudCenDepth--;
        }







//以上地图调整完毕





//获取当前帧对应地图附近的地图点，由于激光雷达的扫描距离为0-150m
//又由于每个cub 的距离是50m , 所以前后左右只需要


		

        int laserCloudValidNum = 0;
        int laserCloudSurroundNum = 0;

		

		//在每一维附近5 个cube (前2个，后2 个，中间1个)里进行查找
		//(前后250m范围内，总共500m 范围)，三个维度总共125个cube





		
		//在这125 个cube 里面进一步筛选在视域范围内的cube		
        for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++) {
          for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) {
            for (int k = centerCubeK - 2; k <= centerCubeK + 2; k++) {
              if (i >= 0 && i < laserCloudWidth && j >= 0 &&
                  j < laserCloudHeight && k >= 0 && k < laserCloudDepth) {
                  //如果索引合法

				  
					//换算成实际比例，在世界坐标系下的坐标
                float centerX = 50.0 * (i - laserCloudCenWidth);
                float centerY = 50.0 * (j - laserCloudCenHeight);
                float centerZ = 50.0 * (k - laserCloudCenDepth);

				//判断是否lidar 视线范围的标志
                bool isInLaserFOV = false;
                for (int ii = -1; ii <= 1; ii += 2) {
                  for (int jj = -1; jj <= 1; jj += 2) {
                    for (int kk = -1; kk <= 1; kk += 2) {
						//上下左右8 个顶点坐标
                      float cornerX = centerX + 25.0 * ii;
                      float cornerY = centerY + 25.0 * jj;
                      float cornerZ = centerZ + 25.0 * kk;
						//原点到顶点距离的平方和
                      float squaredSide1 =
                          (transformTobeMapped[3] - cornerX) *
                              (transformTobeMapped[3] - cornerX) +
                          (transformTobeMapped[4] - cornerY) *
                              (transformTobeMapped[4] - cornerY) +
                          (transformTobeMapped[5] - cornerZ) *
                              (transformTobeMapped[5] - cornerZ);

						//fov 设定的范围为60度,上边30度，下边30度，cos 30 = sqrt(3.0)/2
						//余弦定理cosA = (b*b + c*c - a*a) / (2*b*c)

						//pointOnYAxis 到顶点距离的平方和
					  float squaredSide2 = (pointOnYAxis.x - cornerX) *
                                               (pointOnYAxis.x - cornerX) +
                                           (pointOnYAxis.y - cornerY) *
                                               (pointOnYAxis.y - cornerY) +
                                           (pointOnYAxis.z - cornerZ) *
                                               (pointOnYAxis.z - cornerZ);

					　
						//由于原点到顶点的距离有正负所以才有如下check1, check2
						//|OP|^2 + |OD|^2 - |PD|^2  < 2*|OP||OD|*cos(30)        (OD 为正，即squaredSide1  > 0)
                      float check1 = 100.0 + squaredSide1 - squaredSide2 -
                                     10.0 * sqrt(3.0) * sqrt(squaredSide1);

						//|OP|^2 + |OD|^2 - |PD|^2  <  -2*|OP||OD|*cos(30)       (OD 为负，即squaredSide1  < 0)
                      float check2 = 100.0 + squaredSide1 - squaredSide2 +
                                     10.0 * sqrt(3.0) * sqrt(squaredSide1);

                      if (check1 < 0 && check2 > 0) {
                        isInLaserFOV = true;
                      }
                    }
                  }
                }

				//记住视域范围内的cube 索引，匹配用
                if (isInLaserFOV) {
                  laserCloudValidInd[laserCloudValidNum] =
                      i + laserCloudWidth * j +
                      laserCloudWidth * laserCloudHeight * k;
                  laserCloudValidNum++;
                }
				//记住附近所有cube 的索引，显示用
                laserCloudSurroundInd[laserCloudSurroundNum] =
                    i + laserCloudWidth * j +
                    laserCloudWidth * laserCloudHeight * k;
                laserCloudSurroundNum++;
              }
            }
          }
        }

        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
		//构建特征点地图，查找匹配使用
        for (int i = 0; i < laserCloudValidNum; i++) {
          *laserCloudCornerFromMap +=
              *laserCloudCornerArray[laserCloudValidInd[i]];
          *laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
        }
        int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
        int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();

        int laserCloudCornerStackNum2 = laserCloudCornerStack2->points.size();
        for (int i = 0; i < laserCloudCornerStackNum2; i++) {
          pointAssociateTobeMapped(&laserCloudCornerStack2->points[i],
                                   &laserCloudCornerStack2->points[i]);
        }

        int laserCloudSurfStackNum2 = laserCloudSurfStack2->points.size();
        for (int i = 0; i < laserCloudSurfStackNum2; i++) {
          pointAssociateTobeMapped(&laserCloudSurfStack2->points[i],
                                   &laserCloudSurfStack2->points[i]);
        }

        laserCloudCornerStack->clear();
		//设置滤波对象
        downSizeFilterCorner.setInputCloud(laserCloudCornerStack2);
		//执行滤波处理
        downSizeFilterCorner.filter(*laserCloudCornerStack);
		//获取滤波后体素点尺寸
        int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

        laserCloudSurfStack->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfStack2);
        downSizeFilterSurf.filter(*laserCloudSurfStack);
        int laserCloudSurfStackNum = laserCloudSurfStack->points.size();

        laserCloudCornerStack2->clear();
        laserCloudSurfStack2->clear();





		//用地图在视野范围内的点云建立kd-tree (边沿特征、平面特征)
		//再对当前帧点云和地图中视野内的点云进行配准(同里程计)
		//最后用最小二乘法求出最佳转移矩阵

        if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 100) {
			//构建边沿点的kd-tree
          kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
			//构建平面点的kd-tree
          kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);

			//开始迭代
          for (int iterCount = 0; iterCount < maxIterNum; iterCount++) {
            laserCloudOri->clear();
            coeffSel->clear();

            // find correspondance
            //边沿特征点的匹配
            for (int i = 0; i < laserCloudCornerStackNum; i++) {
              pointOri = laserCloudCornerStack->points[i];
			  //获取该点在地图中的坐标
              pointAssociateToMap(&pointOri, &pointSel);
			  //寻找最近距离最近的5 个点
              kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                                  pointSearchSqDis);

				//5 个点中最大的距离不超过1 才处理
              if (pointSearchSqDis[4] < 1.0) {
			  	//将五个最近点的坐标加和求平均
                float cx = 0;
                float cy = 0;
                float cz = 0;
                for (int j = 0; j < 5; j++) {
                  cx += laserCloudCornerFromMap->points[pointSearchInd[j]].x;
                  cy += laserCloudCornerFromMap->points[pointSearchInd[j]].y;
                  cz += laserCloudCornerFromMap->points[pointSearchInd[j]].z;
                }
                cx /= 5;
                cy /= 5;
                cz /= 5;

				//求均方差
                float a11 = 0;
                float a12 = 0;
                float a13 = 0;
                float a22 = 0;
                float a23 = 0;
                float a33 = 0;
                for (int j = 0; j < 5; j++) {
                  float ax =
                      laserCloudCornerFromMap->points[pointSearchInd[j]].x - cx;
                  float ay =
                      laserCloudCornerFromMap->points[pointSearchInd[j]].y - cy;
                  float az =
                      laserCloudCornerFromMap->points[pointSearchInd[j]].z - cz;

                  a11 += ax * ax;
                  a12 += ax * ay;
                  a13 += ax * az;
                  a22 += ay * ay;
                  a23 += ay * az;
                  a33 += az * az;
                }
                a11 /= 5;
                a12 /= 5;
                a13 /= 5;
                a22 /= 5;
                a23 /= 5;
                a33 /= 5;

				//构建矩阵
                matA1.at<float>(0, 0) = a11;
                matA1.at<float>(0, 1) = a12;
                matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12;
                matA1.at<float>(1, 1) = a22;
                matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13;
                matA1.at<float>(2, 1) = a23;
                matA1.at<float>(2, 2) = a33;

				//特征值分解

                cv::eigen(matA1, matD1, matV1);

				//如果最大的特征值大于第二大的特征值三倍以上
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                  float x0 = pointSel.x;
                  float y0 = pointSel.y;
                  float z0 = pointSel.z;
                  float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                  float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                  float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                  float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                  float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                  float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                  float a012 =
                      sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
                               ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                           ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
                               ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                           ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
                               ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

                  float l12 =
                      sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
                           (z1 - z2) * (z1 - z2));

                  float la =
                      ((y1 - y2) *
                           ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                       (z1 - z2) *
                           ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) /
                      a012 / l12;

                  float lb =
                      -((x1 - x2) *
                            ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
                        (z1 - z2) *
                            ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
                      a012 / l12;

                  float lc =
                      -((x1 - x2) *
                            ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                        (y1 - y2) *
                            ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
                      a012 / l12;

                  float ld2 = a012 / l12;

                  pointProj = pointSel;
                  pointProj.x -= la * ld2;
                  pointProj.y -= lb * ld2;
                  pointProj.z -= lc * ld2;


				  //权重系数

                  float s = 1 - 0.9 * fabs(ld2);

                  coeff.x = s * la;
                  coeff.y = s * lb;
                  coeff.z = s * lc;
                  coeff.intensity = s * ld2;

					//距离足够小才使用
                  if (s > 0.1) {
                    laserCloudOri->push_back(pointOri);
                    coeffSel->push_back(coeff);
                  }
                }
              }
            }
            // find correspondance
            //平面特征点的匹配
            for (int i = 0; i < laserCloudSurfStackNum; i++) {
              pointOri = laserCloudSurfStack->points[i];
              pointAssociateToMap(&pointOri, &pointSel);
              kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                                pointSearchSqDis);

              if (pointSearchSqDis[4] < 1.0) {
			  	//构建五个最近点的坐标矩阵
                for (int j = 0; j < 5; j++) {
                  matA0.at<float>(j, 0) =
                      laserCloudSurfFromMap->points[pointSearchInd[j]].x;
                  matA0.at<float>(j, 1) =
                      laserCloudSurfFromMap->points[pointSearchInd[j]].y;
                  matA0.at<float>(j, 2) =
                      laserCloudSurfFromMap->points[pointSearchInd[j]].z;
                }
				//求解 matA0 * matX0 = matB0
                cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

                float pa = matX0.at<float>(0, 0);
                float pb = matX0.at<float>(1, 0);
                float pc = matX0.at<float>(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                  if (fabs(pa *
                               laserCloudSurfFromMap->points[pointSearchInd[j]]
                                   .x +
                           pb *
                               laserCloudSurfFromMap->points[pointSearchInd[j]]
                                   .y +
                           pc *
                               laserCloudSurfFromMap->points[pointSearchInd[j]]
                                   .z +
                           pd) > 0.2) {
                    planeValid = false;
                    break;
                  }
                }

                if (planeValid) {
                  float pd2 =
                      pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                  pointProj = pointSel;
                  pointProj.x -= pa * pd2;
                  pointProj.y -= pb * pd2;
                  pointProj.z -= pc * pd2;

                  float s =
                      1 -
                      0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x +
                                                  pointSel.y * pointSel.y +
                                                  pointSel.z * pointSel.z));

                  coeff.x = s * pa;
                  coeff.y = s * pb;
                  coeff.z = s * pc;
                  coeff.intensity = s * pd2;

                  if (s > 0.1) {
                    laserCloudOri->push_back(pointOri);
                    coeffSel->push_back(coeff);
                  }
                }
              }
            }

            float srx = sin(transformTobeMapped[0]);
            float crx = cos(transformTobeMapped[0]);
            float sry = sin(transformTobeMapped[1]);
            float cry = cos(transformTobeMapped[1]);
            float srz = sin(transformTobeMapped[2]);
            float crz = cos(transformTobeMapped[2]);

            int laserCloudSelNum = laserCloudOri->points.size();
			//如果特征点太少
            if (laserCloudSelNum < 50) {
              continue;
            }



			//最小二乘法求最佳旋转矩阵

            cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
            cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
            cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
            cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
            for (int i = 0; i < laserCloudSelNum; i++) {
              pointOri = laserCloudOri->points[i];
              coeff = coeffSel->points[i];

              float arx =
                  (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y -
                   srx * sry * pointOri.z) *
                      coeff.x +
                  (-srx * srz * pointOri.x - crz * srx * pointOri.y -
                   crx * pointOri.z) *
                      coeff.y +
                  (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y -
                   cry * srx * pointOri.z) *
                      coeff.z;

              float ary = ((cry * srx * srz - crz * sry) * pointOri.x +
                           (sry * srz + cry * crz * srx) * pointOri.y +
                           crx * cry * pointOri.z) *
                              coeff.x +
                          ((-cry * crz - srx * sry * srz) * pointOri.x +
                           (cry * srz - crz * srx * sry) * pointOri.y -
                           crx * sry * pointOri.z) *
                              coeff.z;

              float arz =
                  ((crz * srx * sry - cry * srz) * pointOri.x +
                   (-cry * crz - srx * sry * srz) * pointOri.y) *
                      coeff.x +
                  (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y +
                  ((sry * srz + cry * crz * srx) * pointOri.x +
                   (crz * sry - cry * srx * srz) * pointOri.y) *
                      coeff.z;

              matA.at<float>(i, 0) = arx;
              matA.at<float>(i, 1) = ary;
              matA.at<float>(i, 2) = arz;
              matA.at<float>(i, 3) = coeff.x;
              matA.at<float>(i, 4) = coeff.y;
              matA.at<float>(i, 5) = coeff.z;
              matB.at<float>(i, 0) = -coeff.intensity;
            }
            cv::transpose(matA, matAt);
            matAtA = matAt * matA;
            matAtB = matAt * matB;
            cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

			//退化场景判断与处理
            if (iterCount == 0) {
              cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
              cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
              cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

              cv::eigen(matAtA, matE, matV);
              matV.copyTo(matV2);

              isDegenerate = false;
              float eignThre[6] = {100, 100, 100, 100, 100, 100};
              for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                  for (int j = 0; j < 6; j++) {
                    matV2.at<float>(i, j) = 0;
                  }
                  isDegenerate = true;
                } else {
                  break;
                }
              }
              matP = matV.inv() * matV2;
            }

            if (isDegenerate) {
              cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
              matX.copyTo(matX2);
              matX = matP * matX2;
            }



			//通过求的最佳转移矩阵修正之前猜测的世界坐标系下的旋转矩阵
			//积累每次的调整量
            transformTobeMapped[0] += matX.at<float>(0, 0);
            transformTobeMapped[1] += matX.at<float>(1, 0);
            transformTobeMapped[2] += matX.at<float>(2, 0);
            transformTobeMapped[3] += matX.at<float>(3, 0);
            transformTobeMapped[4] += matX.at<float>(4, 0);
            transformTobeMapped[5] += matX.at<float>(5, 0);

            float deltaR = sqrt(pow(rad2deg(matX.at<float>(0, 0)), 2) +
                                pow(rad2deg(matX.at<float>(1, 0)), 2) +
                                pow(rad2deg(matX.at<float>(2, 0)), 2));
            float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
                                pow(matX.at<float>(4, 0) * 100, 2) +
                                pow(matX.at<float>(5, 0) * 100, 2));
			//旋转平移量足够小就停止迭代
            if (deltaR < 0.05 && deltaT < 0.05) {
              break;
            }
          }

			//迭代结果更新相关的转移矩阵
          transformUpdate();
        }




		//---------------------------建图----------------


		

		//将corner points 按距离(比例尺缩小) 归入相应的立方体
        for (int i = 0; i < laserCloudCornerStackNum; i++) {
			//转移到世界坐标
          pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);

			//按50 的比例尺缩小，四舍五入，偏移laserCloudCen* 的
			//量，计算索引
			//这里加25.0, 后面在加一个laserCloudCenWidth  实现了四舍五入的效果
          int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
          int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
          int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

			//如果pointSel.x 小于-25，四舍五入的方向就反了，需要减去1
          if (pointSel.x + 25.0 < 0)
            cubeI--;
          if (pointSel.y + 25.0 < 0)
            cubeJ--;
          if (pointSel.z + 25.0 < 0)
            cubeK--;

          if (cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 &&
              cubeJ < laserCloudHeight && cubeK >= 0 &&
              cubeK < laserCloudDepth) {
              //只挑选-laserCloudCenWidth*50.0 < point.x < laserCloudCenWidth * 50.0
              //范围内的点，y 和z 同理按照尺寸放进不同的组，
              //每个组的点数量各异


			  //计算该特征点属于立方体的id
            int cubeInd = cubeI + laserCloudWidth * cubeJ +
                          laserCloudWidth * laserCloudHeight * cubeK;
			  //把特征点加入到立方体中
            laserCloudCornerArray[cubeInd]->push_back(pointSel);
          }
        }


		

		//将surf points按距离(比例尺缩小)归入相应的立方体
        for (int i = 0; i < laserCloudSurfStackNum; i++) {
			//转移到世界坐标系
          pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

			//计算平面特征点对于的立方体
          int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
          int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
          int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;


			////如果pointSel.x 小于-25，四舍五入的方向就反了，需要减去1
          if (pointSel.x + 25.0 < 0)
            cubeI--;
          if (pointSel.y + 25.0 < 0)
            cubeJ--;
          if (pointSel.z + 25.0 < 0)
            cubeK--;


			//把特征点加入地图中
          if (cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 &&
              cubeJ < laserCloudHeight && cubeK >= 0 &&
              cubeK < laserCloudDepth) {
              //计算特征点属于立方体的id
            int cubeInd = cubeI + laserCloudWidth * cubeJ +
                          laserCloudWidth * laserCloudHeight * cubeK;
			  //把特征点加入地图对应的立方体
            laserCloudSurfArray[cubeInd]->push_back(pointSel);
          }
        }


			//特征点下采样

		//对地图中的每个cub 立方体进行下采样
        for (int i = 0; i < laserCloudValidNum; i++) {
          int ind = laserCloudValidInd[i];

          laserCloudCornerArray2[ind]->clear();
          downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
          downSizeFilterCorner.filter(*laserCloudCornerArray2[ind]);

          laserCloudSurfArray2[ind]->clear();
          downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
          downSizeFilterSurf.filter(*laserCloudSurfArray2[ind]);
			//Array 与Array2 交换，即滤波后自我更新
          pcl::PointCloud<PointType>::Ptr laserCloudTemp =
              laserCloudCornerArray[ind];
          laserCloudCornerArray[ind] = laserCloudCornerArray2[ind];
          laserCloudCornerArray2[ind] = laserCloudTemp;

          laserCloudTemp = laserCloudSurfArray[ind];
          laserCloudSurfArray[ind] = laserCloudSurfArray2[ind];
          laserCloudSurfArray2[ind] = laserCloudTemp;
        }
		//地图帧数加1
        mapFrameCount++;




		
		//特征点汇总下采样，每隔五帧publish 一次，从第一次开始
        if (mapFrameCount >= mapFrameNum) {
          mapFrameCount = 0;

          laserCloudSurround2->clear();


			//对 地图的  边沿特征点和平面特征点进行汇总
          #ifdef OUTPUT_ALL_POINTCLOUD
          for (int i = 0; i < laserCloudNum; i++) {
            *laserCloudSurround2 += *laserCloudCornerArray[i];
            *laserCloudSurround2 += *laserCloudSurfArray[i];
          }
          #else
          for (int i = 0; i < laserCloudSurroundNum; i++) {
            int ind = laserCloudSurroundInd[i];
            *laserCloudSurround2 += *laserCloudCornerArray[ind];
            *laserCloudSurround2 += *laserCloudSurfArray[ind];
          }
          #endif

			//对汇总的 特征点进行下采样，
			
          laserCloudSurround->clear();
          downSizeFilterCorner.setInputCloud(laserCloudSurround2);
          downSizeFilterCorner.filter(*laserCloudSurround);

			//发布地图下采样后汇总的特征点
			//包括边沿特征和平面特征
          sensor_msgs::PointCloud2 laserCloudSurround3;
          pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
          laserCloudSurround3.header.stamp =
              ros::Time().fromSec(timeLaserOdometry);
          laserCloudSurround3.header.frame_id = "/camera_init";
          pubLaserCloudSurround.publish(laserCloudSurround3);

          // mappingCloudFreq.report();
        }


			//将点云中全部点转移到世界坐标系下
        int laserCloudFullResNum = laserCloudFullRes->points.size();
        for (int i = 0; i < laserCloudFullResNum; i++) {
          pointAssociateToMap(&laserCloudFullRes->points[i],
                              &laserCloudFullRes->points[i]);
        }

		//发布时间坐标系下对应的当前帧全部特征点
        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp =
            ros::Time().fromSec(timeLaserOdometry);
        laserCloudFullRes3.header.frame_id = "/camera_init";
        pubLaserCloudFullRes.publish(laserCloudFullRes3);

        geometry_msgs::Quaternion geoQuat =
            tf::createQuaternionMsgFromRollPitchYaw(transformAftMapped[2],
                                                    -transformAftMapped[0],
                                                    -transformAftMapped[1]);

        odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
        odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
        odomAftMapped.pose.pose.orientation.z = geoQuat.x;
        odomAftMapped.pose.pose.orientation.w = geoQuat.w;
        odomAftMapped.pose.pose.position.x = transformAftMapped[3];
        odomAftMapped.pose.pose.position.y = transformAftMapped[4];
        odomAftMapped.pose.pose.position.z = transformAftMapped[5];
		//扭转量
        odomAftMapped.twist.twist.angular.x = transformBefMapped[0];
        odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
        odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
        odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
        odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
        odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
        pubOdomAftMapped.publish(odomAftMapped);

        // mappingOdometryFreq.report();
		//广播坐标系旋转平移参量
        aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
        aftMappedTrans.setRotation(
            tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        aftMappedTrans.setOrigin(tf::Vector3(transformAftMapped[3],
                                             transformAftMapped[4],
                                             transformAftMapped[5]));
        tfBroadcaster.sendTransform(aftMappedTrans);
      }
    }

    status = ros::ok();
    rate.sleep();
  }

  return 0;
}
