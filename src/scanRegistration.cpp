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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

#include <loam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
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

using std::sin;
using std::cos;
using std::atan2;

#ifndef VELODYNE_HDL64E
const double scanPeriod = 0.1; // time duration per scan
#else
const double scanPeriod = 0.1; // TODO
#endif

const int systemDelay = 20;
int systemInitCount = 0;
bool systemInited = false;

#ifndef VELODYNE_HDL64E
const int N_SCANS = 16; /////
#else
const int N_SCANS = 64;
#endif

#ifndef VELODYNE_HDL64E
const int MAX_POINTS = 40000;
#else
const int MAX_POINTS = 160000;
#endif

float cloudCurvature[MAX_POINTS];
int cloudSortInd[MAX_POINTS];
int cloudNeighborPicked[MAX_POINTS];
int cloudLabel[MAX_POINTS];

int imuPointerFront = 0;
int imuPointerLast = -1;

#ifndef VELODYNE_HDL64E
const int imuQueLength = 200;
#else
const int imuQueLength = 2000;
#endif

float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
float imuRollCur = 0, imuPitchCur = 0, imuYawCur = 0;

float imuVeloXStart = 0, imuVeloYStart = 0, imuVeloZStart = 0;
float imuShiftXStart = 0, imuShiftYStart = 0, imuShiftZStart = 0;

float imuVeloXCur = 0, imuVeloYCur = 0, imuVeloZCur = 0;
float imuShiftXCur = 0, imuShiftYCur = 0, imuShiftZCur = 0;

float imuShiftFromStartXCur = 0, imuShiftFromStartYCur = 0,
      imuShiftFromStartZCur = 0; // updated by ShiftToStartIMU()
float imuVeloFromStartXCur = 0, imuVeloFromStartYCur = 0,
      imuVeloFromStartZCur = 0; // updated by VeloToStartIMU()

double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};
float imuYaw[imuQueLength] = {0};

float imuAccX[imuQueLength] = {0};
float imuAccY[imuQueLength] = {0};
float imuAccZ[imuQueLength] = {0};
// updated by AccumulateIMUShift()
float imuVeloX[imuQueLength] = {0};
float imuVeloY[imuQueLength] = {0};
float imuVeloZ[imuQueLength] = {0};
// updated by AccumulateIMUShift()
float imuShiftX[imuQueLength] = {0};
float imuShiftY[imuQueLength] = {0};
float imuShiftZ[imuQueLength] = {0};

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubImuTrans;

// imu shift from start vector (imuShiftFromStart*Cur) converted into start imu
// coordinates?
//计算局部坐标系下点云中的点相对第一个开始点的由于加减速运动产生的位移畸变
void ShiftToStartIMU(float pointTime) {
//计算相对于第一个点由于加减速产生的畸变位移,该位移是在全球坐标系中
//(全局坐标系下畸变位移量 delta_Tg)
  imuShiftFromStartXCur =
      imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
  imuShiftFromStartYCur =
      imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
  imuShiftFromStartZCur =
      imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;

//下面是将畸变位移从全局坐标系转换到机体坐标系
//前面我们从机体坐标系转到全局坐标系R = Ry(yaw)*Rx(pitch)*Rz(roll)
//现在需要从世界坐标系到机体坐标系，就需要用R 的转置矩阵乘以当前delta_Tg

//先y 轴旋转(-imuYawStart), 即Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuShiftFromStartXCur -
             sin(imuYawStart) * imuShiftFromStartZCur;
  float y1 = imuShiftFromStartYCur;
  float z1 = sin(imuYawStart) * imuShiftFromStartXCur +
             cos(imuYawStart) * imuShiftFromStartZCur;
//绕x 轴旋转(-imuPitchStart), 即Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;
//绕z 轴旋转(-imuRollStart), 即Rz(roll).inverse
  imuShiftFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuShiftFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuShiftFromStartZCur = z2;
}
// imu velocity from start vector (imuVeloFromStart*Cur) converted into start
// imu coordinates?
//计算局部坐标系下点云中的点相对于第一个开始点由于加减速产生的速度畸变
void VeloToStartIMU() {
//计算相对于第一个点由于加减速产生的畸变速度，该速度为全局坐标系
  imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
  imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
  imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;
//把畸变速度转换到机体坐标系

//绕y 轴旋转(-imuYawStart), 即Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuVeloFromStartXCur -
             sin(imuYawStart) * imuVeloFromStartZCur;
  float y1 = imuVeloFromStartYCur;
  float z1 = sin(imuYawStart) * imuVeloFromStartXCur +
             cos(imuYawStart) * imuVeloFromStartZCur;
//绕x 轴旋转(-imuPitchStart), 即Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;
//绕Z 轴旋转(-imuRollStart), 即Rz(roll).inverse
  imuVeloFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuVeloFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuVeloFromStartZCur = z2;
}
// points converted into start imu coordinates?
//去除点云加减速产生的位移畸变
//因为位移的畸变量是在第一个点云出的机体坐标系下产生的
//所以需要把当前点云根据当前姿态旋转到世界坐标系下，
//再通过开始点云的姿态把把旋转到开始点云对应的机体坐标系下
//再加上畸变量
void TransformToStartIMU(PointType *p) {
//通过该点云的当前姿态把点云旋转到全局坐标

//绕Z 轴旋转(imuRollCur)
  float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
  float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
  float z1 = p->z;

//绕x 轴旋转(imupitchCurr)
  float x2 = x1;
  float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
  float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;
//绕y 轴旋转(imuYawCur)
  float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
  float y3 = y2;
  float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

//通过开始点云的姿态把该点云全局坐标旋转到开始点云对应的机体坐标系，
//再加上畸变位移，就实现了开始点云后面的所有的点云都移位到开始点云的机体坐标系下

//绕y 轴 旋转(-imuYawStart)
  float x4 = cos(imuYawStart) * x3 - sin(imuYawStart) * z3;
  float y4 = y3;
  float z4 = sin(imuYawStart) * x3 + cos(imuYawStart) * z3;
//绕x 轴 旋转(-imuPitchStart)
  float x5 = x4;
  float y5 = cos(imuPitchStart) * y4 + sin(imuPitchStart) * z4;
  float z5 = -sin(imuPitchStart) * y4 + cos(imuPitchStart) * z4;

//绕z 轴旋转(-imuRollStart), 然后叠加平移
  p->x =
      cos(imuRollStart) * x5 + sin(imuRollStart) * y5 + imuShiftFromStartXCur;
  p->y =
      -sin(imuRollStart) * x5 + cos(imuRollStart) * y5 + imuShiftFromStartYCur;
  p->z = z5 + imuShiftFromStartZCur;
}
// compute last shift to imuShift*[imuPointerLast] and velo to
// imuVelo*[imuPointerLast] using previous shift/velo/acc
//通过求出的zxy 各个轴的加速度，积分速度和位移
void AccumulateIMUShift() {
//获取世界坐标系下的姿态
  float roll = imuRoll[imuPointerLast];
  float pitch = imuPitch[imuPointerLast];
  float yaw = imuYaw[imuPointerLast];

//或者zxy 坐标系下的真实加速度
  float accX = imuAccX[imuPointerLast];
  float accY = imuAccY[imuPointerLast];
  float accZ = imuAccZ[imuPointerLast];


//将当前时刻的加速度值绕交换过的ZXY固定轴(原XYZ) 分别旋转(roll, pitch, yaw)角度
//转换得到世界走坐标系下的加速度值

//绕Z 轴旋转(roll)
  float x1 = cos(roll) * accX - sin(roll) * accY;
  float y1 = sin(roll) * accX + cos(roll) * accY;
  float z1 = accZ;

//绕X 轴旋转(pitch)
  float x2 = x1;
  float y2 = cos(pitch) * y1 - sin(pitch) * z1;
  float z2 = sin(pitch) * y1 + cos(pitch) * z1;

//绕Y 轴旋转(yaw)
  accX = cos(yaw) * x2 + sin(yaw) * z2;
  accY = y2;
  accZ = -sin(yaw) * x2 + cos(yaw) * z2;

  int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;
  double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];
  if (timeDiff < scanPeriod) {

    imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] +
                                imuVeloX[imuPointerBack] * timeDiff +
                                accX * timeDiff * timeDiff / 2;
    imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] +
                                imuVeloY[imuPointerBack] * timeDiff +
                                accY * timeDiff * timeDiff / 2;
    imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] +
                                imuVeloZ[imuPointerBack] * timeDiff +
                                accZ * timeDiff * timeDiff / 2;

    imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;
    imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
    imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;
  }
}

auto last_time = std::chrono::system_clock::now();


//接收点云数据，velodyne 雷达坐标系安装为x 轴向前，y 轴向左，z 轴向上的右手坐标系
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg) {

//一包数据是包括了激光雷达旋转一周采集到的所有数据
  if (!systemInited) {//丢弃前20 个点云数据
    systemInitCount++;
    if (systemInitCount >= systemDelay) {
      systemInited = true;
    }
    return;
  }

//记录每个scan 有曲率的点的开始和结束索引
  std::vector<int> scanStartInd(
      N_SCANS, 0); // scanStartInd[scanId] is the first point id of scanId
  std::vector<int> scanEndInd(
      N_SCANS, 0); // scanEndInd[scanId] is the last point id of scanId

//当前点云时间
  double timeScanCur =
      laserCloudMsg->header.stamp.toSec(); // time point of current scan
  pcl::PointCloud<pcl::PointXYZ>
      laserCloudIn; // input cloud, NaN points removed
//消息转换成pcl 数据存放
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
  std::vector<int> indices;
  //移除空点
  pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);

  //ROS_INFO("cloud recieved");
  if (false) {
    // write clound to file
    static bool written = false;
    if (!written) {
      std::ofstream ofs("/home/i-yanghao/tmp/normalized_cloud.xyz");
      if (ofs) {
        for (int i = 0; i < laserCloudIn.points.size(); i++) {
          auto & p = laserCloudIn.points[i];
          float len = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
          ofs << p.x / len << " " << p.y / len << " " << p.z / len << std::endl;
        }
        ROS_INFO("cloud written");
        written = true;
      }
    }
  }

//点云点的数量
  int cloudSize = laserCloudIn.points.size(); // number of cloud points

  //lidar scan 开始点的旋转角，atan2范围[-pi, +pi], 计算旋转角是取负号是因为
  //velodyne 是顺时针旋转的
  float startOri =
      -atan2(laserCloudIn.points[0].y,
             laserCloudIn.points[0]
                 .x); // ori of first point in cloud on origin x-y plane

//lidar scan 结束点的旋转角，加2*pi 使点云旋转周期为2 * pi
  float endOri =
      -atan2(laserCloudIn.points[cloudSize - 1]
                 .y, // ori of last point in clound on origin x-y plane
             laserCloudIn.points[cloudSize - 1].x) +
      2 * M_PI;

//结束方位角与开始方位角差值控制在(PI, 3*PI) 范围，允许lidar 不是一个圆周扫描
//正常情况下在这个范围内: pi < endOri - startOri < 3*pi, 异常则修正
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }

//lidar 扫描是否旋转过半
//这里是检测一个一组数据包中的一个点云的位置，是否旋转过半
  bool halfPassed = false;
  int count = cloudSize;
  PointType point;
  std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);

  // float minAngle = 180, maxAngle = -180;
  // PointType minP, maxP;
  // minP.x = minP.y = minP.z = 1e8;
  // maxP.x = maxP.y = maxP.z = -1e8;

  /// use imu data to register original scanned points into lidar coodinates in
  /// different scan lines
  //利用imu 数据将原始扫描点配准到激光雷达坐标系中
  //就把每个点都统一到开始点云的机体坐标系中
  //目的是去除激光雷达非匀速运动产生的畸变
  //激光雷达扫描一圈需要时间，如果这个时候激光雷达相对地面有相对运动
  //就会导致后面的点云跟第一个点云机体坐标有一定的畸变
  for (int i = 0; i < cloudSize; i++) {
  	//坐标轴交换，velodyne lidar的坐标系也转换到z 轴向前，x 轴向左的右手坐标系
    point.x = laserCloudIn.points[i].y;
    point.y = laserCloudIn.points[i].z;
    point.z = laserCloudIn.points[i].x;

    // minP.x = std::min(minP.x, point.x);
    // minP.y = std::min(minP.y, point.y);
    // minP.z = std::min(minP.z, point.z);
    // maxP.x = std::max(maxP.x, point.x);
    // maxP.y = std::max(maxP.y, point.y);
    // maxP.z = std::max(maxP.z, point.z);

	//计算点的仰角(根据lidar 文档垂直角计算公式)，根据仰角排列激光线号，velodyne
	//每两个scan 之间间隔2 度
    float angle = atan(point.y / sqrt(point.x * point.x + point.z * point.z)) *
                  180 / M_PI; // angle of origin z from origin x-y plane
    int scanID;
    // if(!std::isnan(angle)) {
    //   minAngle = std::min(angle, minAngle);
    //   maxAngle = std::max(angle, maxAngle);
    // }
    // ROS_INFO("[%f]", angle);

    // compute scanID
#ifndef VELODYNE_HDL64E
	//仰角四舍五入(加减0.5 截断效果等于四舍五入)
    int roundedAngle = int(angle + (angle < 0.0 ? -0.5 : +0.5));

	//把该点云放到16 线上去
    if (roundedAngle > 0) {
      scanID = roundedAngle;
    } else {
      scanID = roundedAngle + (N_SCANS - 1);
    }
#else
    const float angleLowerBoundDeg = -24.8f;
    const float angleUpperBoundDeg = 2.0f;
    const float angleSpan = angleUpperBoundDeg - angleLowerBoundDeg;
    const float angleStep = angleSpan / (N_SCANS - 1);
    float angleID = (angle - angleLowerBoundDeg) / angleStep;
    
    scanID = int(angleID + 0.5f);
#endif

	//过滤点，只挑选[-15度，+15度] 范围内的点，scanID 属于[0, 15]
    if (scanID > (N_SCANS - 1) || scanID < 0) { // drop the points with invalid scanIDs
      count--;
      continue;
    }

    const int debug_errorPointIDStart = 121513;
    // if (i >= debug_errorPointIDStart) {
    //   ROS_INFO("point %i's scanID = %i", i, scanID);
    // }    

	//该点的旋转角
    float ori = -atan2(point.x, point.z);

	//根据扫描线是否旋转过半选择与起始位置还是终止位置进行差值计算，从而进行补偿
    if (!halfPassed) {
	//确保 -pi/2 < ori - startOri < 3*pi/2
      if (ori < startOri - M_PI / 2) {
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }

      if (ori - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      ori += 2 * M_PI;

	//确保 -3*pi/2 < ori - endOri < pi/2
      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      }
    }

	//-0.5 < relTime < 1.5, 点旋转的角度与整个周期旋转角度的比率，即点云中点的相对时间
    float relTime = (ori - startOri) / (endOri - startOri);
	//点强度= 线号+ 点相对时间(即一个整数 +  一个小数，整数部分是线号，小数部分是该点
	//的相对时间)， 匀速扫描:根据当前扫描的角度和扫描的周期计算相对扫描起始位置的时间

	//可以理解成整数部分是仰角，小数部分是旋转角和周期的比例
	point.intensity = scanID + scanPeriod * relTime;

    // if (i >= debug_errorPointIDStart) {
    //   ROS_INFO("halfPassed = %i, ori = %f, point intensity = %f", halfPassed,
    //            ori, point.intensity);
    // }

	//点时间= 点云时间+ 周期时间
	//如果收到imu 数据，使用IMU 矫正点云畸变
    if (imuPointerLast >= 0) {
	//计算点的周期时间
      float pointTime = relTime * scanPeriod;
	//寻找是否有点云的时间戳小于IMU 的时间戳的IMU位置:imuPointerFront
      while (imuPointerFront != imuPointerLast) {
        if (timeScanCur + pointTime < imuTime[imuPointerFront]) {
          break;
        }
		//在环形buffer 中获取下一个点的index
        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
      }

      if (timeScanCur + pointTime >
          imuTime[imuPointerFront]) { /// use the newest imu data if no newer
                                      /// imu
        //没找到，此时imuPointerFront==imtPointerLast, 只能以当前收到的最新的imu 的速度，位移，欧拉
        //角作为当前点的速度，位移，欧拉角使用
        imuRollCur = imuRoll[imuPointerFront];
        imuPitchCur = imuPitch[imuPointerFront];
        imuYawCur = imuYaw[imuPointerFront];

        imuVeloXCur = imuVeloX[imuPointerFront];
        imuVeloYCur = imuVeloY[imuPointerFront];
        imuVeloZCur = imuVeloZ[imuPointerFront];

        imuShiftXCur = imuShiftX[imuPointerFront];
        imuShiftYCur = imuShiftY[imuPointerFront];
        imuShiftZCur = imuShiftZ[imuPointerFront];
      } else { /// interpolate in all existing imu data if there are newer imu
               /// data
       //找到了点云时间戳小于imu 时间戳的IMU位置，则该点必处于imuPointerBack
       //和imuPointerFront 之间，据此线性插值，计算点云的速度，位移和欧拉角

		//获取比imuPointerFront 早的一份imu
        int imuPointerBack =
            (imuPointerFront + imuQueLength - 1) % imuQueLength;
		//按时间距离计算权重分配比率， 也即时线性插值
        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) /
                           (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) /
                          (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

        imuRollCur = imuRoll[imuPointerFront] * ratioFront +
                     imuRoll[imuPointerBack] * ratioBack;
        imuPitchCur = imuPitch[imuPointerFront] * ratioFront +
                      imuPitch[imuPointerBack] * ratioBack;
        if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront +
                      (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
        } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront +
                      (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
        } else {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront +
                      imuYaw[imuPointerBack] * ratioBack;
        }

        imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront +
                      imuVeloX[imuPointerBack] * ratioBack;
        imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront +
                      imuVeloY[imuPointerBack] * ratioBack;
        imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront +
                      imuVeloZ[imuPointerBack] * ratioBack;

        imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront +
                       imuShiftX[imuPointerBack] * ratioBack;
        imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront +
                       imuShiftY[imuPointerBack] * ratioBack;
        imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront +
                       imuShiftZ[imuPointerBack] * ratioBack;
      }
      if (i == 0) {
	  	//如果是第一点，记住点云起始位置的速度，位移，欧拉角
        imuRollStart = imuRollCur;
        imuPitchStart = imuPitchCur;
        imuYawStart = imuYawCur;

        imuVeloXStart = imuVeloXCur;
        imuVeloYStart = imuVeloYCur;
        imuVeloZStart = imuVeloZCur;

        imuShiftXStart = imuShiftXCur;
        imuShiftYStart = imuShiftYCur;
        imuShiftZStart = imuShiftZCur;
      } else {
      //计算之后每个点相对于第一个点的由于加减速费匀速运动产生的位移速度畸变，
      //并对点云中的每个点位置信息重新补偿矫正
        ShiftToStartIMU(pointTime);
        VeloToStartIMU();
        TransformToStartIMU(&point);
      }
    }

	//将每个补偿矫正的点放入对应线号的容器中
    laserCloudScans[scanID].push_back(point);
  }

  //ROS_INFO("all points are grouped");

  // ROS_INFO("\n");
  // ROS_INFO("minAngle = %f, maxAngle = %f\n", minAngle, maxAngle);
  // output minAngle = -15, maxAngle = 15
  // ROS_INFO("bounding box = [%f,%f,%f; %f,%f,%f]\n", minP.x, minP.y, minP.z,
  // maxP.x, maxP.y, maxP.z);
  // output generally: [-20(+-10), -5(+-1), -100(+-20); +70(+-10), +25(+-1),
  // +80(+-10)]

//获得有效范围内的点的数量
  cloudSize = count;

  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  //将所有的点按照线号从小到大放入一个容器
  for (int i = 0; i < N_SCANS; i++) {
    *laserCloud += laserCloudScans[i];
  }
  int scanCount = -1;
  //使用每个点的前后五个点计算曲率，因此前五个点和后五个点跳过
  for (int i = 5; i < cloudSize - 5; i++) {
    //ROS_INFO("i = %i, cloundSize = %i", i, cloudSize);
    float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x +
                  laserCloud->points[i - 3].x + laserCloud->points[i - 2].x +
                  laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x +
                  laserCloud->points[i + 1].x + laserCloud->points[i + 2].x +
                  laserCloud->points[i + 3].x + laserCloud->points[i + 4].x +
                  laserCloud->points[i + 5].x;
    float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y +
                  laserCloud->points[i - 3].y + laserCloud->points[i - 2].y +
                  laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y +
                  laserCloud->points[i + 1].y + laserCloud->points[i + 2].y +
                  laserCloud->points[i + 3].y + laserCloud->points[i + 4].y +
                  laserCloud->points[i + 5].y;
    float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z +
                  laserCloud->points[i - 3].z + laserCloud->points[i - 2].z +
                  laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z +
                  laserCloud->points[i + 1].z + laserCloud->points[i + 2].z +
                  laserCloud->points[i + 3].z + laserCloud->points[i + 4].z +
                  laserCloud->points[i + 5].z;
	//曲率计算
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
	//记录曲率点的索引
    cloudSortInd[i] = i;
	//初始时，点全未筛选过
    cloudNeighborPicked[i] = 0;
	//初始化为less flat 点
    cloudLabel[i] = 0;

	//每个scan(线号), 只有第一个符合的点会进来，因为每个scan 的点都放在一起存放
	//这里取的是整数部分就是点强度的线号
    if (int(laserCloud->points[i].intensity) != scanCount) {
	//控制每个scan 只进入第一个点,  这里取的是
      scanCount = int(laserCloud->points[i].intensity);

		//曲率只取同一个scan (线号) 计算出来的，跨scan 计算的曲率非法，
		//排除，也即排除每个scan 的前后五个点
      if (scanCount > 0 && scanCount < N_SCANS) {
        scanStartInd[scanCount] = i + 5;
        scanEndInd[scanCount - 1] = i - 5;
      }
    }
  }
  //第一个scan 曲率点有效点序从第5 个开始最后一个激光线结束点序 size -5
  scanStartInd[0] = 5;
  scanEndInd.back() = cloudSize - 5;

  //ROS_INFO("cloudCurvature scanStartInd scanEndInd computed");



//挑选点，排除容易被斜面挡住的点以及离群点，有些点容易被斜面挡住
//而离群点可能出现有偶然性，这些情况都可能导致前后两次扫描不能被同时看到

  for (int i = 5; i < cloudSize - 6; i++) {//与后一个点的差值，所以减6
    float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
    float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
    float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
	//计算有效曲率点与后一个点之间的距离平方和
    float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;

    if (diff > 0.1) {//前提两个点之间距离要大于0.1

	//点的深度
      float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x +
                          laserCloud->points[i].y * laserCloud->points[i].y +
                          laserCloud->points[i].z * laserCloud->points[i].z);
	//后一个点的深度
      float depth2 =
          sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x +
               laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
               laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);

	//按照两个点的深度比例，将深度较大的点拉回来后计算距离
	//这里的意思是把长的边投影到短的边上
      if (depth1 > depth2) {
        diffX = laserCloud->points[i + 1].x -
                laserCloud->points[i].x * depth2 / depth1;
        diffY = laserCloud->points[i + 1].y -
                laserCloud->points[i].y * depth2 / depth1;
        diffZ = laserCloud->points[i + 1].z -
                laserCloud->points[i].z * depth2 / depth1;

		//边长比也即时弧度值，若小于0.1， 说明夹角比较小，斜面比较陡峭，点深度变化比较
		//剧烈，点处在近似与激光束平行的斜面上
        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 <
            0.1) { // is connected?
         //排除容易被斜面挡住的点
         //该点及前面五个点( 大致都在斜面上 )  全部置为筛选过
          cloudNeighborPicked[i - 5] = 1;
          cloudNeighborPicked[i - 4] = 1;
          cloudNeighborPicked[i - 3] = 1;
          cloudNeighborPicked[i - 2] = 1;
          cloudNeighborPicked[i - 1] = 1;
          cloudNeighborPicked[i] = 1;
        }
      } else {
        diffX = laserCloud->points[i + 1].x * depth1 / depth2 -
                laserCloud->points[i].x;
        diffY = laserCloud->points[i + 1].y * depth1 / depth2 -
                laserCloud->points[i].y;
        diffZ = laserCloud->points[i + 1].z * depth1 / depth2 -
                laserCloud->points[i].z;

        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 <
            0.1) { // is connected?
          //排除容易被斜面挡住的点
         //该点及前面六个点 ( 大致都在斜面上 )  全部置为筛选过
          cloudNeighborPicked[i + 1] = 1;
          cloudNeighborPicked[i + 2] = 1;
          cloudNeighborPicked[i + 3] = 1;
          cloudNeighborPicked[i + 4] = 1;
          cloudNeighborPicked[i + 5] = 1;
          cloudNeighborPicked[i + 6] = 1;
        }
      }
    }

	//当前点和前一点的距离差
    float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
    float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
    float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
	//距离差的平方
    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

	//当前点的深度的平方
    float dis = laserCloud->points[i].x * laserCloud->points[i].x +
                laserCloud->points[i].y * laserCloud->points[i].y +
                laserCloud->points[i].z * laserCloud->points[i].z;

	//与前后点的平方和都大于深度平方和的万分之二，这些点视为离群点，包括陡斜面上
	//的点，强烈凹凸点和空旷区域中的某些点，设置为筛选过，弃用
    if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) {
      cloudNeighborPicked[i] = 1;
    }
  }

  //ROS_INFO("cloudNeighborPicked initialized");

  pcl::PointCloud<PointType> cornerPointsSharp;     // the outputs
  pcl::PointCloud<PointType> cornerPointsLessSharp; // the outputs
  pcl::PointCloud<PointType> surfPointsFlat;        // the outputs
  pcl::PointCloud<PointType> surfPointsLessFlat;    // the outputs

	//将每条线上的点分入相应的类别: 边沿点和平面点特征点
  for (int i = 0; i < N_SCANS; i++) {
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(
        new pcl::PointCloud<PointType>);
	//将每个scan 的曲率点分成6 等份处理，确保周围都有点被选作
    for (int j = 0; j < 6; j++) {
		//六等份的起点: sp = scanStartInd + (scanEndInd - scanStartInd) * j / 6
      int sp = (scanStartInd[i] * (6 - j) + scanEndInd[i] * j) / 6;
		//六等份的终点: ep = scanStartInd -1 + (scanEndInd - scanStartInd)*(j+1) / 6 -1
      int ep = (scanStartInd[i] * (5 - j) + scanEndInd[i] * (j + 1)) / 6 - 1;

	//按曲率从小到大冒泡排序
      for (int k = sp + 1; k <= ep; k++) { // sort by curvature within [sp,
                                           // ep]?, curvature descending order
        for (int l = k; l >= sp + 1; l--) {
			//如果后面曲率大于前面则交换
          if (cloudCurvature[cloudSortInd[l]] <
              cloudCurvature[cloudSortInd[l - 1]]) {
            int temp = cloudSortInd[l - 1];
            cloudSortInd[l - 1] = cloudSortInd[l];
            cloudSortInd[l] = temp;
          }
        }
      }

	//挑选每个分段的曲率很大和比较大的点
      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSortInd[k];//曲率最大点的点序
        //如果曲率大的点，曲率的确比较大，并且未被筛选过滤掉
        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > 0.1) {

          largestPickedNum++;
		  //挑选曲率最大的前2 个点放入sharp 点集合
          if (largestPickedNum <= 2) {
            cloudLabel[ind] = 2;// 2 代表曲率很大
            cornerPointsSharp.push_back(laserCloud->points[ind]);
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else if (largestPickedNum <= 20) {
          //挑选曲率最大的前20 个点放入less sharp 点集合
            cloudLabel[ind] = 1;// 1 代表曲率比较尖锐
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else {
            break;
          }

		//筛选标志置位
          cloudNeighborPicked[ind] = 1;

		//将曲率比较大的点的前后各5 个连续距离比较近的点筛选出去，防止特征点聚集，
		//使得特征点在每个方向上尽量分布均匀
          for (int l = 1; l <= 5; l++) {
            float diffX = laserCloud->points[ind + l].x -
                          laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y -
                          laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z -
                          laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x -
                          laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y -
                          laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z -
                          laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }
	//挑选每个分段的曲率很小比较小的点
      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];// 曲率最小点的点序
        //如果曲率的确比较小，并且未被筛选过
        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0.1) {

          cloudLabel[ind] = -1;//-1 代表曲率很小的点
          surfPointsFlat.push_back(laserCloud->points[ind]);

          smallestPickedNum++;
		  //只选最小的四个，剩余的label = 0, 就是曲率比较小的
          if (smallestPickedNum >= 4) {
            break;
          }
		//筛选标志设置
          cloudNeighborPicked[ind] = 1;

		//同样防止特征点聚集
          for (int l = 1; l <= 5; l++) {
            float diffX = laserCloud->points[ind + l].x -
                          laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y -
                          laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z -
                          laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x -
                          laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y -
                          laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z -
                          laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

	//将剩余的点(包括之前被排除的点) 全部归入平面点中 less plat 类别中
      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0) {
          surfPointsLessFlatScan->push_back(laserCloud->points[k]);
        }
      }
    }

	//由于less flat 点最多，对每个分段less flat 的点进行体素栅格滤波
    pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
	//声明体素滤波器的对象
    pcl::VoxelGrid<PointType> downSizeFilter;
	//加入需要滤波的点云
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
	//体素栅格大小设置为，20*20*20cm
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(surfPointsLessFlatScanDS);

	//less flat 点汇总
    surfPointsLessFlat += surfPointsLessFlatScanDS;
  }

  //ROS_INFO("feature points collected");

//publish 消除非匀速运动畸变后的所有点
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/camera";
  pubLaserCloud.publish(laserCloudOutMsg);

//publish 消除非匀速运动畸变后的平面点和边沿点
  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/camera";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);

  sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
  pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
  cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsLessSharpMsg.header.frame_id = "/camera";
  pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/camera";
  pubSurfPointsFlat.publish(surfPointsFlat2);

  sensor_msgs::PointCloud2 surfPointsLessFlat2;
  pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
  surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsLessFlat2.header.frame_id = "/camera";
  pubSurfPointsLessFlat.publish(surfPointsLessFlat2);
//publish imu 消息，由于循环到了最后，因此是cur 都是代表最后一个点
//即最后一个点的欧拉角，畸变位移及一个点云周期增加的速度
  pcl::PointCloud<pcl::PointXYZ> imuTrans(4, 1);
//起始点的欧拉角
  imuTrans.points[0].x = imuPitchStart;
  imuTrans.points[0].y = imuYawStart;
  imuTrans.points[0].z = imuRollStart;

//最后一个点的欧拉角
  imuTrans.points[1].x = imuPitchCur;
  imuTrans.points[1].y = imuYawCur;
  imuTrans.points[1].z = imuRollCur;
//最后一个点相对于第一个点的畸变位移和速度
  imuTrans.points[2].x = imuShiftFromStartXCur;
  imuTrans.points[2].y = imuShiftFromStartYCur;
  imuTrans.points[2].z = imuShiftFromStartZCur;

  imuTrans.points[3].x = imuVeloFromStartXCur;
  imuTrans.points[3].y = imuVeloFromStartYCur;
  imuTrans.points[3].z = imuVeloFromStartZCur;

  sensor_msgs::PointCloud2 imuTransMsg;
  pcl::toROSMsg(imuTrans, imuTransMsg);
  imuTransMsg.header.stamp = laserCloudMsg->header.stamp;
  imuTransMsg.header.frame_id = "/camera";
  pubImuTrans.publish(imuTransMsg);

  // #define PRINT(name) ROS_INFO("in scanRegistration "#name" = %f", name)
  //  PRINT(imuShiftFromStartXCur);
  //  PRINT(imuShiftFromStartYCur);
  //  PRINT(imuShiftFromStartZCur);
  //  PRINT(imuVeloFromStartXCur);
  //  PRINT(imuVeloFromStartYCur);
  //  PRINT(imuVeloFromStartZCur);
  // #undef PRINT
}

void imuHandler(const sensor_msgs::Imu::ConstPtr &imuIn) {
  //ROS_INFO("imu recieved!\n");
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(imuIn->orientation, orientation);
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

	//机体坐标系消除重力影响
  float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;
  float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
  float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

  //#define PRINT(name) ROS_INFO(#name" = %f\n", name)
  //  PRINT(accX);
  //  PRINT(accY);
  //  PRINT(accZ);
  //#undef PRINT

//循环移位效果，形成环形数组
  imuPointerLast = (imuPointerLast + 1) % imuQueLength;

//记录imu 时间戳和姿态(世界坐标系下)
  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = roll;
  imuPitch[imuPointerLast] = pitch;
  imuYaw[imuPointerLast] = yaw;

  //记录出去重力影响后，zxy 坐标系各个方向的加速度
  imuAccX[imuPointerLast] = accX;
  imuAccY[imuPointerLast] = accY;
  imuAccZ[imuPointerLast] = accZ;

  AccumulateIMUShift();
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "scanRegistration");
  ros::NodeHandle nh;

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(
      "/velodyne_points", 2, laserCloudHandler);

  ros::Subscriber subImu =
      nh.subscribe<sensor_msgs::Imu>("/imu/data", 50, imuHandler);

  pubLaserCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 2);

  pubCornerPointsSharp =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 2);

  pubCornerPointsLessSharp =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 2);

  pubSurfPointsFlat =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 2);

  pubSurfPointsLessFlat =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 2);

  pubImuTrans = nh.advertise<sensor_msgs::PointCloud2>("/imu_trans", 5);

  ros::spin();

  return 0;
}
