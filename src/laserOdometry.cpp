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
#include <cmath>

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

//一个点云周期
const float scanPeriod = 0.1;


#ifndef VELODYNE_HDL64E
//跳帧数，控制发给laserMapping 的频率
const int skipFrameNum = 1;
const int maxIterNum = 25;
#else
const int skipFrameNum = 1;
const int maxIterNum = 100;
#endif

bool systemInited = false;

//时间戳信息
double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;
double timeImuTrans = 0;

//消息接收标志
bool newCornerPointsSharp = false;
bool newCornerPointsLessSharp = false;
bool newSurfPointsFlat = false;
bool newSurfPointsLessFlat = false;
bool newLaserCloudFullRes = false;
bool newImuTrans = false;

pcl::PointCloud<PointType>::Ptr
    cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    surfPointsLessFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    laserCloudOri(new pcl::PointCloud<PointType>()); // stores the original
                                                     // coordinates (not
                                                     // transformed to start
                                                     // time point) of feature
                                                     // point in current cloud
pcl::PointCloud<PointType>::Ptr coeffSel(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<pcl::PointXYZ>::Ptr
    imuTrans(new pcl::PointCloud<pcl::PointXYZ>());
pcl::KdTreeFLANN<PointType>::Ptr
    kdtreeCornerLast(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr
    kdtreeSurfLast(new pcl::KdTreeFLANN<PointType>());

int laserCloudCornerLastNum;
int laserCloudSurfLastNum;

#ifndef VELODYNE_HDL64E
const int MAX_POINTS = 40000;
#else
const int MAX_POINTS = 160000;
#endif

int pointSelCornerInd[MAX_POINTS];
float pointSearchCornerInd1[MAX_POINTS];
float pointSearchCornerInd2[MAX_POINTS];

int pointSelSurfInd[MAX_POINTS];
float pointSearchSurfInd1[MAX_POINTS];
float pointSearchSurfInd2[MAX_POINTS];
float pointSearchSurfInd3[MAX_POINTS];
//当前帧相对于上一帧的状态转移量
float transform[6] = {
    0}; // rotation {x, y, z} or {pitch?, yaw?, roll?}, translation {x, y, z}
//当前帧相对于第一帧的状态转移量
float transformSum[6] = {
    0}; // rotation {x, y, z} or {pitch?, yaw?, roll?}, translation {x, y, z}
//点云第一个点的rpy
float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
//点云最后一个点的rpy
float imuRollLast = 0, imuPitchLast = 0, imuYawLast = 0;
//点云最后一个点相对于第一个点由于加减速产生的畸变位移
float imuShiftFromStartX = 0, imuShiftFromStartY = 0, imuShiftFromStartZ = 0;
//点云最后一个点相对于第一个点由于加减速产生的畸变速度
float imuVeloFromStartX = 0, imuVeloFromStartY = 0, imuVeloFromStartZ = 0;

//将当前帧点云TransformTostart 和将上一帧点云TransformToEnd 的作用:
//取出畸变，并将两帧点云数据统一到通一个坐标系下计算


//当前点云中的点相对第一个点去除因匀速运动产生的畸变，效果相当于得到在
//点云扫描开始位置静止扫描得到的点云
void TransformToStart(PointType const *const pi, PointType *const po) {
//插值系数计算，点云中每个点的相对时间/ 点云周期
  float s = 10 * (pi->intensity - int(pi->intensity)); // interpolation factor

//线性插值: 根据每个点在点云中的相对位置关系，乘以相应的旋转平移系数
  float rx = s * transform[0];
  float ry = s * transform[1];
  float rz = s * transform[2];
  float tx = s * transform[3];
  float ty = s * transform[4];
  float tz = s * transform[5];

	//平移后绕z 轴旋转(-rz)
  float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
  float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
  float z1 = (pi->z - tz);

	//绕x 轴旋转(-rx)
  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;

	//绕y 轴旋转
  po->x = cos(ry) * x2 - sin(ry) * z2;
  po->y = y2;
  po->z = sin(ry) * x2 + cos(ry) * z2;
  po->intensity = pi->intensity;
}


//将上一帧点云中的点相对于结束位置去除因匀速运动产生的畸变，效果
//相当于得到在点云扫描结束位置静止扫描得到的点云

void TransformToEnd(PointType const *const pi, PointType *const po) {

//插值系数计算
  float s = 10 * (pi->intensity - int(pi->intensity)); // interpolation factor

	//线性插值
  float rx = s * transform[0];
  float ry = s * transform[1];
  float rz = s * transform[2];
  float tx = s * transform[3];
  float ty = s * transform[4];
  float tz = s * transform[5];

  //这里同TransformToStart 一样，求出相对于起始点校准的坐标
  //这里主要校准匀速运动产生的畸变

//平移后绕z 轴旋转(-rz)
  float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
  float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
  float z1 = (pi->z - tz);

	//绕x 轴旋转(-rx)
  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;
	//绕y轴旋转(-ry)
  float x3 = cos(ry) * x2 - sin(ry) * z2;
  float y3 = y2;
  float z3 = sin(ry) * x2 + cos(ry) * z2;

  //根据当前里程计transform, 即当前姿态，把前一帧的中的点云转到世界坐标系中

  rx = transform[0];
  ry = transform[1];
  rz = transform[2];
  tx = transform[3];
  ty = transform[4];
  tz = transform[5];

  //绕y 轴旋转(ry)

  float x4 = cos(ry) * x3 + sin(ry) * z3;
  float y4 = y3;
  float z4 = -sin(ry) * x3 + cos(ry) * z3;

//绕x 轴旋转(rx)
  float x5 = x4;
  float y5 = cos(rx) * y4 - sin(rx) * z4;
  float z5 = sin(rx) * y4 + cos(rx) * z4;
//绕 z 轴旋转(rz), 再平移
  float x6 = cos(rz) * x5 - sin(rz) * y5 + tx;
  float y6 = sin(rz) * x5 + cos(rz) * y5 + ty;
  float z6 = z5 + tz;

  //这里是把前一帧中的点云在世界坐标系中的坐标根据
  //开始点云的姿态转换到开始点云的坐标系中

//绕z 轴旋转(imuRollStart)， 同时平移了点云最后一点相对于第一个点的匀速位移
	//imuShiftFromStartX, 点云最后一个点相对于第一个点在世界坐标系中加减速产生的位移
  float x7 = cos(imuRollStart) * (x6 - imuShiftFromStartX) -
             sin(imuRollStart) * (y6 - imuShiftFromStartY);
  float y7 = sin(imuRollStart) * (x6 - imuShiftFromStartX) +
             cos(imuRollStart) * (y6 - imuShiftFromStartY);
  float z7 = z6 - imuShiftFromStartZ;

//绕x  轴旋转(imuPitchStart)
  float x8 = x7;
  float y8 = cos(imuPitchStart) * y7 - sin(imuPitchStart) * z7;
  float z8 = sin(imuPitchStart) * y7 + cos(imuPitchStart) * z7;
//绕y 轴旋转(imuyawStart)
  float x9 = cos(imuYawStart) * x8 + sin(imuYawStart) * z8;
  float y9 = y8;
  float z9 = -sin(imuYawStart) * x8 + cos(imuYawStart) * z8;


//根据点云最后一个点的姿态，把该点云旋转到最后一个点的机体坐标系中
//该点和最后一个点云的姿态都是相对于开始点云的姿态的，所以这里
//可以把开始点云理解为该点云和最后一个点云转换的中间坐标系，类似于世界
//坐标系相对于两帧不同的点云的姿态转换一样
  
//绕y 轴旋转(-imuYawStart)
  float x10 = cos(imuYawLast) * x9 - sin(imuYawLast) * z9;
  float y10 = y9;
  float z10 = sin(imuYawLast) * x9 + cos(imuYawLast) * z9;
//绕x 轴旋转(-imuPitchLast)
  float x11 = x10;
  float y11 = cos(imuPitchLast) * y10 + sin(imuPitchLast) * z10;
  float z11 = -sin(imuPitchLast) * y10 + cos(imuPitchLast) * z10;
// 绕z 轴旋转(-imuRollLast)
  po->x = cos(imuRollLast) * x11 + sin(imuRollLast) * y11;
  po->y = -sin(imuRollLast) * x11 + cos(imuRollLast) * y11;
  po->z = z11;
  //只保留线号
  po->intensity = int(pi->intensity);
}


//利用IMU  修正旋转量，根据起始欧拉角，当前点云的欧拉角修正


void PluginIMURotation(float bcx, float bcy, float bcz, float blx, float bly,
                       float blz, float alx, float aly, float alz, float &acx,
                       float &acy, float &acz) {
  float sbcx = sin(bcx);
  float cbcx = cos(bcx);
  float sbcy = sin(bcy);
  float cbcy = cos(bcy);
  float sbcz = sin(bcz);
  float cbcz = cos(bcz);

  float sblx = sin(blx);
  float cblx = cos(blx);
  float sbly = sin(bly);
  float cbly = cos(bly);
  float sblz = sin(blz);
  float cblz = cos(blz);

  float salx = sin(alx);
  float calx = cos(alx);
  float saly = sin(aly);
  float caly = cos(aly);
  float salz = sin(alz);
  float calz = cos(alz);

  float srx = -sbcx * (salx * sblx + calx * caly * cblx * cbly +
                       calx * cblx * saly * sbly) -
              cbcx * cbcz * (calx * saly * (cbly * sblz - cblz * sblx * sbly) -
                             calx * caly * (sbly * sblz + cbly * cblz * sblx) +
                             cblx * cblz * salx) -
              cbcx * sbcz * (calx * caly * (cblz * sbly - cbly * sblx * sblz) -
                             calx * saly * (cbly * cblz + sblx * sbly * sblz) +
                             cblx * salx * sblz);
  acx = -asin(srx);

  float srycrx = (cbcy * sbcz - cbcz * sbcx * sbcy) *
                     (calx * saly * (cbly * sblz - cblz * sblx * sbly) -
                      calx * caly * (sbly * sblz + cbly * cblz * sblx) +
                      cblx * cblz * salx) -
                 (cbcy * cbcz + sbcx * sbcy * sbcz) *
                     (calx * caly * (cblz * sbly - cbly * sblx * sblz) -
                      calx * saly * (cbly * cblz + sblx * sbly * sblz) +
                      cblx * salx * sblz) +
                 cbcx * sbcy * (salx * sblx + calx * caly * cblx * cbly +
                                calx * cblx * saly * sbly);
  float crycrx = (cbcz * sbcy - cbcy * sbcx * sbcz) *
                     (calx * caly * (cblz * sbly - cbly * sblx * sblz) -
                      calx * saly * (cbly * cblz + sblx * sbly * sblz) +
                      cblx * salx * sblz) -
                 (sbcy * sbcz + cbcy * cbcz * sbcx) *
                     (calx * saly * (cbly * sblz - cblz * sblx * sbly) -
                      calx * caly * (sbly * sblz + cbly * cblz * sblx) +
                      cblx * cblz * salx) +
                 cbcx * cbcy * (salx * sblx + calx * caly * cblx * cbly +
                                calx * cblx * saly * sbly);
  acy = atan2(srycrx / cos(acx), crycrx / cos(acx));

  float srzcrx = sbcx * (cblx * cbly * (calz * saly - caly * salx * salz) -
                         cblx * sbly * (caly * calz + salx * saly * salz) +
                         calx * salz * sblx) -
                 cbcx * cbcz * ((caly * calz + salx * saly * salz) *
                                    (cbly * sblz - cblz * sblx * sbly) +
                                (calz * saly - caly * salx * salz) *
                                    (sbly * sblz + cbly * cblz * sblx) -
                                calx * cblx * cblz * salz) +
                 cbcx * sbcz * ((caly * calz + salx * saly * salz) *
                                    (cbly * cblz + sblx * sbly * sblz) +
                                (calz * saly - caly * salx * salz) *
                                    (cblz * sbly - cbly * sblx * sblz) +
                                calx * cblx * salz * sblz);
  float crzcrx = sbcx * (cblx * sbly * (caly * salz - calz * salx * saly) -
                         cblx * cbly * (saly * salz + caly * calz * salx) +
                         calx * calz * sblx) +
                 cbcx * cbcz * ((saly * salz + caly * calz * salx) *
                                    (sbly * sblz + cbly * cblz * sblx) +
                                (caly * salz - calz * salx * saly) *
                                    (cbly * sblz - cblz * sblx * sbly) +
                                calx * calz * cblx * cblz) -
                 cbcx * sbcz * ((saly * salz + caly * calz * salx) *
                                    (cblz * sbly - cbly * sblx * sblz) +
                                (caly * salz - calz * salx * saly) *
                                    (cbly * cblz + sblx * sbly * sblz) -
                                calx * calz * cblx * sblz);
  acz = atan2(srzcrx / cos(acx), crzcrx / cos(acx));
}

//相对于第一个点云即原点，积累旋转量
void AccumulateRotation(float cx, float cy, float cz, float lx, float ly,
                        float lz, float &ox, float &oy, float &oz) {
  //求s2
  float srx = cos(lx) * cos(cx) * sin(ly) * sin(cz) -
              cos(cx) * cos(cz) * sin(lx) - cos(lx) * cos(ly) * sin(cx);
  //求2 对应的角度
  ox = -asin(srx);

	//求c2s3
  float srycrx =
      sin(lx) * (cos(cy) * sin(cz) - cos(cz) * sin(cx) * sin(cy)) +
      cos(lx) * sin(ly) * (cos(cy) * cos(cz) + sin(cx) * sin(cy) * sin(cz)) +
      cos(lx) * cos(ly) * cos(cx) * sin(cy);
	//求c2c3
  float crycrx =
      cos(lx) * cos(ly) * cos(cx) * cos(cy) -
      cos(lx) * sin(ly) * (cos(cz) * sin(cy) - cos(cy) * sin(cx) * sin(cz)) -
      sin(lx) * (sin(cy) * sin(cz) + cos(cy) * cos(cz) * sin(cx));
	//求出3对应的角度
  oy = atan2(srycrx / cos(ox), crycrx / cos(ox));

	//求c2s1
  float srzcrx =
      sin(cx) * (cos(lz) * sin(ly) - cos(ly) * sin(lx) * sin(lz)) +
      cos(cx) * sin(cz) * (cos(ly) * cos(lz) + sin(lx) * sin(ly) * sin(lz)) +
      cos(lx) * cos(cx) * cos(cz) * sin(lz);
	//c1c2
  float crzcrx =
      cos(lx) * cos(lz) * cos(cx) * cos(cz) -
      cos(cx) * sin(cz) * (cos(ly) * sin(lz) - cos(lz) * sin(lx) * sin(ly)) -
      sin(cx) * (sin(ly) * sin(lz) + cos(ly) * cos(lz) * sin(lx));
	//求出1 对应的角度
  oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox));
}

void laserCloudSharpHandler(
    const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2) {
  timeCornerPointsSharp = cornerPointsSharp2->header.stamp.toSec();

  cornerPointsSharp->clear();
  pcl::fromROSMsg(*cornerPointsSharp2, *cornerPointsSharp);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cornerPointsSharp, *cornerPointsSharp, indices);
  newCornerPointsSharp = true;
}

void laserCloudLessSharpHandler(
    const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2) {
  timeCornerPointsLessSharp = cornerPointsLessSharp2->header.stamp.toSec();

  cornerPointsLessSharp->clear();
  pcl::fromROSMsg(*cornerPointsLessSharp2, *cornerPointsLessSharp);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cornerPointsLessSharp, *cornerPointsLessSharp,
                               indices);
  newCornerPointsLessSharp = true;
}

void laserCloudFlatHandler(
    const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2) {
  timeSurfPointsFlat = surfPointsFlat2->header.stamp.toSec();

  surfPointsFlat->clear();
  pcl::fromROSMsg(*surfPointsFlat2, *surfPointsFlat);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*surfPointsFlat, *surfPointsFlat, indices);
  newSurfPointsFlat = true;
}

void laserCloudLessFlatHandler(
    const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2) {
  timeSurfPointsLessFlat = surfPointsLessFlat2->header.stamp.toSec();

  surfPointsLessFlat->clear();
  pcl::fromROSMsg(*surfPointsLessFlat2, *surfPointsLessFlat);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*surfPointsLessFlat, *surfPointsLessFlat,
                               indices);
  newSurfPointsLessFlat = true;
}

void laserCloudFullResHandler(
    const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2) {
  timeLaserCloudFullRes = laserCloudFullRes2->header.stamp.toSec();

  laserCloudFullRes->clear();
  pcl::fromROSMsg(*laserCloudFullRes2, *laserCloudFullRes);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*laserCloudFullRes, *laserCloudFullRes, indices);
  newLaserCloudFullRes = true;
}

void imuTransHandler(const sensor_msgs::PointCloud2ConstPtr &imuTrans2) {
  timeImuTrans = imuTrans2->header.stamp.toSec();

  imuTrans->clear();
  pcl::fromROSMsg(*imuTrans2, *imuTrans);

  imuPitchStart = imuTrans->points[0].x;
  imuYawStart = imuTrans->points[0].y;
  imuRollStart = imuTrans->points[0].z;

  imuPitchLast = imuTrans->points[1].x;
  imuYawLast = imuTrans->points[1].y;
  imuRollLast = imuTrans->points[1].z;

  imuShiftFromStartX = imuTrans->points[2].x;
  imuShiftFromStartY = imuTrans->points[2].y;
  imuShiftFromStartZ = imuTrans->points[2].z;

  imuVeloFromStartX = imuTrans->points[3].x;
  imuVeloFromStartY = imuTrans->points[3].y;
  imuVeloFromStartZ = imuTrans->points[3].z;

  newImuTrans = true;
}

FreqReport laserOdometryFreq("laserOdometry");
FreqReport registeredLaserCloudFreq("registeredLaserCloud");
int main(int argc, char **argv) {
  ros::init(argc, argv, "laserOdometry");
  ros::NodeHandle nh;

  ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>(
      "/laser_cloud_sharp", 2, laserCloudSharpHandler);

  ros::Subscriber subCornerPointsLessSharp =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 2,
                                             laserCloudLessSharpHandler);

  ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>(
      "/laser_cloud_flat", 2, laserCloudFlatHandler);

  ros::Subscriber subSurfPointsLessFlat =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 2,
                                             laserCloudLessFlatHandler);

  ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>(
      "/velodyne_cloud_2", 2, laserCloudFullResHandler);

  ros::Subscriber subImuTrans =
      nh.subscribe<sensor_msgs::PointCloud2>("/imu_trans", 5, imuTransHandler);

  ros::Publisher pubLaserCloudCornerLast =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);

  ros::Publisher pubLaserCloudSurfLast =
      nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);

  ros::Publisher pubLaserCloudFullRes =
      nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 2);

  ros::Publisher pubLaserOdometry =
      nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 5);
  nav_msgs::Odometry laserOdometry;
  laserOdometry.header.frame_id = "/camera_init";
  laserOdometry.child_frame_id = "/laser_odom";

  tf::TransformBroadcaster tfBroadcaster;
  tf::StampedTransform laserOdometryTrans;
  laserOdometryTrans.frame_id_ = "/camera_init";
  laserOdometryTrans.child_frame_id_ = "/laser_odom";

//搜索到的点序
  std::vector<int> pointSearchInd;
//搜索到的点平方距离
  std::vector<float> pointSearchSqDis;

  // pointOri stores the original coordinates (not transformed to start time
  // point) of feature point in current cloud
  // coeff.xyz stores step * diff(distance(pointSel, {edge/plane}), {pointSel.x,
  // pointSel.y, pointSel.z}), diff means gradient
  // coeff.instance stores step * distance(pointSel, {edge/plane})
  PointType pointOri, pointSel, //选中的特征点
  tripod1, tripod2, tripod3,// 特征点对应的点
      /*pointProj, */ coeff;

//退化标志
  bool isDegenerate = false;
//P 矩阵，预测矩阵，协方差矩阵
  cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

  int frameCount = skipFrameNum;
  ros::Rate rate(100);
  bool status = ros::ok();
  while (status) {
    ros::spinOnce();

	//同步作用，确保同时收到同一个点云的特征点以及IMU 信息才进入
    if (newCornerPointsSharp && newCornerPointsLessSharp && newSurfPointsFlat &&
        newSurfPointsLessFlat && newLaserCloudFullRes && newImuTrans &&
        fabs(timeCornerPointsSharp - timeSurfPointsLessFlat) < 0.005 &&
        fabs(timeCornerPointsLessSharp - timeSurfPointsLessFlat) < 0.005 &&
        fabs(timeSurfPointsFlat - timeSurfPointsLessFlat) < 0.005 &&
        fabs(timeLaserCloudFullRes - timeSurfPointsLessFlat) < 0.005 &&
        fabs(timeImuTrans - timeSurfPointsLessFlat) < 0.005) {
      newCornerPointsSharp = false;
      newCornerPointsLessSharp = false;
      newSurfPointsFlat = false;
      newSurfPointsLessFlat = false;
      newLaserCloudFullRes = false;
      newImuTrans = false;

	//将第一个点云数据集发送给laserMapping, 从下一个点云数据开始处理
      if (!systemInited) {
        // initialize the "last clouds" & kdtrees, publish the first clouds,
        // initialize the pitch and roll components of transformSum

        // initialize laserCloudCornerLast
        //将cornerPointsLessSharp 与 laserCloudCornerLast 交换，目的保存cornerPointsLessSharp 的值下轮使用
        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;
        // initialize laserCloudSurfLast
        //将surfPointLessFlat 与laserCloudSurfLast 交换，目的保存surfPointsLessFlat 的值下轮使用
        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;
        // initialize kdtreeCornerLast kdtreeSurfLast
        //使用上一帧的特征点构建kd-tree
        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);  //所有的边沿点集合
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast); //所有的平面点集合
        // publish the first laserCloudCornerLast
        //将cornerPointsLessSharp 和surfPointLessFlat 点也即边沿点和平面点分别发给laserMapping
        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp =
            ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);
        // publish the first laserCloudSurfLast2
        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp =
            ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

		//记住原点的pitch 和roll
        transformSum[0] += imuPitchStart; // TODO
        transformSum[2] += imuRollStart;  // TODO

        systemInited = true;
        continue;
      }

      // minus the predicted motion
      //T 平移量的初值为加减速的位移量，为其梯度下降的方向
      //沿用上次转换的T,同时在其基础上减去匀速运动的位移，即只考虑加减速的位移量
      transform[3] -= imuVeloFromStartX * scanPeriod;
      transform[4] -= imuVeloFromStartY * scanPeriod;
      transform[5] -= imuVeloFromStartZ * scanPeriod;

      if (laserCloudCornerLastNum > 10 &&
          laserCloudSurfLastNum >
              100) { // when features are sufficient in last cloud
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cornerPointsSharp, *cornerPointsSharp,
                                     indices);

        int cornerPointsSharpNum = cornerPointsSharp->points.size();
        int surfPointsFlatNum = surfPointsFlat->points.size();

		//Levenberg-Marquardt算法(L-M method), 非线性最小二乘算法，最优化算法的一种
        for (int iterCount = 0; iterCount < maxIterNum; iterCount++) {
          laserCloudOri->clear();
          coeffSel->clear();


//---------------角特征点中计算边与边的残差-----------------------
//------re2e = |(Pw  - P5)x(Pw - P1)|  /  |P5 - P1|----------------
//其实就是计算点到线的距离，分子的叉乘计算的是三个点组成的三角形的面积，所以计算的
//就是三角形的高，即点到线的距离

		  
		//处理当前点云的曲率最大的特征点，从上一个点云中曲率比较大的特征点中
		//找两个最近距离点，一个点使用kd-tree 查找，另一个根据找到的点在其相邻线找另外一个最近
		//距离的点
          for (int i = 0; i < cornerPointsSharpNum; i++) {
            // transform current point to the frame of start time point
            //去除当前帧中的点相对于开始点的匀速运动畸变
            TransformToStart(&cornerPointsSharp->points[i], &pointSel);

			//每次迭代五次，重新查找最近点
            if (iterCount % 5 == 0) { // locate the nearest point in last corner
                                      // points to this cornerPointsSharp point
                                      // every 5 iters
              std::vector<int> indices;
              pcl::removeNaNFromPointCloud(*laserCloudCornerLast,
                                           *laserCloudCornerLast, indices);
			  //kd-tree 查找一个最近距离点，边沿未经过体素栅格滤波，一般边沿点本来就比较
			  //少，不做滤波
              kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd,
                                               pointSearchSqDis);
			  //寻找相邻线距离目标点距离最小的点
			  
              int closestPointInd = -1, minPointInd2 = -1;
			  //最小距离的平方必须要小于25
              if (pointSearchSqDis[0] < 25) { // when distance to the found
                                              // nearest point pointSearchInd[0]
                                              // in last cloud be < 5
                //获取最近点的id
                closestPointInd = pointSearchInd[0];

				//提取最近点的线号
                int closestPointScan = int(
                    laserCloudCornerLast->points[closestPointInd]
                        .intensity); // get the scan line id of closestPointInd

				//初始门槛值为5 米，可大致过滤掉scan ID 相邻，但实际线不相邻
				//的值
                float pointSqDis, minPointSqDis2 = 25; // max distance: 5
                //寻找距离目标点最近距离的平方和最小的点
                for (int j = closestPointInd + 1; j < cornerPointsSharpNum;
                     j++) { // search forward points, find the other point in
                            // last cloud
                  // TODO: j is bounded by cornerPointsSharpNum, can it be used
                  // to index laserCloudCornerLast?
                  //非相邻线
                  if (int(laserCloudCornerLast->points[j].intensity) >
                      closestPointScan + 2.5) {
                    break;
                  }
                  // squared distance between the other point and pointSel
                  pointSqDis =
                      (laserCloudCornerLast->points[j].x - pointSel.x) *
                          (laserCloudCornerLast->points[j].x - pointSel.x) +
                      (laserCloudCornerLast->points[j].y - pointSel.y) *
                          (laserCloudCornerLast->points[j].y - pointSel.y) +
                      (laserCloudCornerLast->points[j].z - pointSel.z) *
                          (laserCloudCornerLast->points[j].z - pointSel.z);

					//确保两个点不在同一条scan 上
                  if (int(laserCloudCornerLast->points[j].intensity) >
                      closestPointScan) { // the other point must not be in the
                                          // same scan
                    //距离更近要小于初始值5米
                    if (pointSqDis < minPointSqDis2) {
						//更新最小距离于点序
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  }
                }

				//非相邻线查找
                for (int j = closestPointInd - 1; j >= 0;
                     j--) { // search backward points
                     //向scanID 减小的方向查找
                  if (int(laserCloudCornerLast->points[j].intensity) <
                      closestPointScan - 2.5) {
                    break;
                  }
                  // squared distance between the other point and pointSel
                  pointSqDis =
                      (laserCloudCornerLast->points[j].x - pointSel.x) *
                          (laserCloudCornerLast->points[j].x - pointSel.x) +
                      (laserCloudCornerLast->points[j].y - pointSel.y) *
                          (laserCloudCornerLast->points[j].y - pointSel.y) +
                      (laserCloudCornerLast->points[j].z - pointSel.z) *
                          (laserCloudCornerLast->points[j].z - pointSel.z);

					//确保两个点不在同一个scan 线上
                  if (int(laserCloudCornerLast->points[j].intensity) <
                      closestPointScan) { // the other point must not be in the
                                          // same scan
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  }
                }
              }

				//记住组成的点序
				//kd-tree 最近距离点，-1 表示未找到满足的点
              pointSearchCornerInd1[i] =
                  closestPointInd; // the first point in last cloud closest to
                                   // pointSel (distance < 5)
               //另一个最近的，-1 表示没有找到满足的点
              pointSearchCornerInd2[i] =
                  minPointInd2; // the second point in neaby scan within last
                                // clound closest to pointSel (distance < 5)
            }

			
			//大于等于0， 不等于-1 , 说明两个点都找到了
            if (pointSearchCornerInd2[i] >= 0) { // found all two closest points
              tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
              tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]];

			//选择的特征点记为O, kd-tree 最近距离的点记为
			//A, 另一个最近距离点记为B
			//这里O 点是当前帧中的一个特征点，A, B 是O 点在上一帧中对应的
			//距离最近的两个点，现在是要求O 到 AB 的距离，也即是三角形OAB,
			//AB 边上的高


              float x0 = pointSel.x;
              float y0 = pointSel.y;
              float z0 = pointSel.z;
              float x1 = tripod1.x;
              float y1 = tripod1.y;
              float z1 = tripod1.z;
              float x2 = tripod2.x;
              float y2 = tripod2.y;
              float z2 = tripod2.z;

              // cross(pointSel - tripod1, pointSel - tripod2) =
              //  [(y0 - y1) (z0 - z2) - (y0 - y2) (z0 - z1),
              //   (x0 - x2) (z0 - z1) - (x0 - x1) (z0 - z2),
              //   (x0 - x1) (y0 - y2) - (x0 - x2) (y0 - y1)]
              // a012 = |cross(pointSel - tripod1, pointSel - tripod2)|
              //求三角形的面积a012 / 2, 也就是，OA 向量和OB 向量的叉积
				//a012 = |OA x OB| 的平方再开方, 这里其实是平面四边形的面积
				//三角形的面积需要除以2
              float a012 =
                  sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *   // z*z    k*k
                           ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +

			  			//y*y    j*j
                       ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
                           ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + 

			  			//x*x   i*i
                       ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
                           ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

              // l12 = |tripod1 - tripod2|
              //这里求是三角形底边AB 的边长， |AB| = sqrt(x*x + y*y + z*z)
              float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
                               (z1 - z2) * (z1 - z2));


				//求三角形O 到AB 边的高的方向向量， 设高与AB 交于点C
				//向量OA x OB 的方向垂直于平面OAB, 平行于OAB 的法向量
				//OAB平面的法向量n = OAxOB / |OAxOB|
				//由于OC(三角形的高) 垂直 向量AB 和 n, 所以向量AB 的单位向量和向量n
				//叉乘就是OC 的单位向量
				//nd = (OAxOB)xAB/(|OAxOB|*|AB|)
              // diff(ld2, x0), ld2 is the distance from pointSel to edge
              // (tripod1, tripod2) as defined below
              // 求x 轴的分量i
              float la =
                  ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                   (z1 - z2) *
                       ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) /
                  a012 / l12;
              // diff(ld2, y0)
              //求y 轴的分量j
              float lb = -((x1 - x2) *
                               ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
                           (z1 - z2) * ((y0 - y1) * (z0 - z2) -
                                        (y0 - y2) * (z0 - z1))) /
                         a012 / l12;
              // diff(ld2, z0)
              //求z 轴的分量k
              float lc = -((x1 - x2) *
                               ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                           (y1 - y2) * ((y0 - y1) * (z0 - z2) -
                                        (y0 - y2) * (z0 - z1))) /
                         a012 / l12;


			//点到直线的距离，即O  到AB 的距离
			//d = |OA x OB| / |AB|
              float ld2 =
                  a012 /
                  l12; // distance from pointSel to edge (tripod1, tripod2)

              // pointProj = pointSel;
              // pointProj.x -= la * ld2;
              // pointProj.y -= lb * ld2;
              // pointProj.z -= lc * ld2;

			//权重计算，距离越大权重越小，距离越小权重越大，得到的权重范围<=1
              float s = 1; // TODO: step? weight?
              if (iterCount >= 5) {  // 5次迭代之后开始增加权重因素
                s = 1 - 1.8 * fabs(ld2); // TODO: why adjust s like this?
              }

			//考虑权重
              coeff.x = s * la;
              coeff.y = s * lb;
              coeff.z = s * lc;
              coeff.intensity = s * ld2;

			//只保留权重大的，也即距离比较小的，同时也舍弃距离为0 的
              if (s > 0.1 && ld2 != 0) { // apply this correspondence only when
                                         // s is not too small and distance is
                                         // not zero
                laserCloudOri->push_back(cornerPointsSharp->points[i]);
                coeffSel->push_back(coeff);
              }
            }
          }


//-------------通过平面特征计算面到面的残差---------------------
//-----------rp2p = (pw-p1)T ((p3-p5)x(p3-p1)) / |(p3 - p5)x(p3 - p1)|-------------------
//分子后边的叉乘部分还是地图三个点组成的三角形的面积，再点乘一得到四面体的体积，
//再除以底面积得到四面体的高，即点到平面的距离



			//对于本次接收到的曲率最小的点，从上次接收到的点云曲率比较小的点中找三点组成平面，
			//一个使用kd-tree 查找，另一个在同一线上查找满足要求的，第三个在不同线上查找满足要求的点
          for (int i = 0; i < surfPointsFlatNum; i++) {
		  	//去除当前帧中点云相对于开始点的匀速畸变
            TransformToStart(&surfPointsFlat->points[i], &pointSel);

            if (iterCount % 5 ==
                0) { // time to find correspondence with planar patches
                //在上一帧的平面点的kd-tree 中查找该平面特征点距离最近的距离
              kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd,
                                             pointSearchSqDis);
              int closestPointInd = -1, minPointInd2 = -1,
                  minPointInd3 = -1; // another point closest to closestPointInd
                                     // [minPointInd2->within] /
                                     // [minPointInd3->not within] same scan
              if (pointSearchSqDis[0] < 25) { // when distance to the found
                                              // nearest point pointSearchInd[0]
                                              // in last cloud be < 5
                closestPointInd = pointSearchInd[0];
                int closestPointScan = int(
                    laserCloudSurfLast->points[closestPointInd]
                        .intensity); // get the scan line id of closestPointInd

                float pointSqDis, minPointSqDis2 = 25, minPointSqDis3 = 25;
                for (int j = closestPointInd + 1; j < surfPointsFlatNum;
                     j++) { // search forward points, find the other point in
                            // last cloud
                  // TODO: j is bounded by surfPointsFlatNum, can it be used to
                  // index laserCloudSurfLast?
                  if (int(laserCloudSurfLast->points[j].intensity) >
                      closestPointScan + 2.5) { // search another point within
                                                // 2.5 scans from pointSel
                    break;
                  }
                  // squared distance between the other point and pointSel
                  pointSqDis =
                      (laserCloudSurfLast->points[j].x - pointSel.x) *
                          (laserCloudSurfLast->points[j].x - pointSel.x) +
                      (laserCloudSurfLast->points[j].y - pointSel.y) *
                          (laserCloudSurfLast->points[j].y - pointSel.y) +
                      (laserCloudSurfLast->points[j].z - pointSel.z) *
                          (laserCloudSurfLast->points[j].z - pointSel.z);
				//如果点的线号小于等于最近点的线号，
				//应该最多取等，也即同一线上的点
                  if (int(laserCloudSurfLast->points[j].intensity) <=
                      closestPointScan) { // within same scan
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  } else { // not within same scan
                  //如果点处在大于该线上
                    if (pointSqDis < minPointSqDis3) {
                      minPointSqDis3 = pointSqDis;
                      minPointInd3 = j;
                    }
                  }
                }
                for (int j = closestPointInd - 1; j >= 0;
                     j--) { // search backward points, find the other point in
                            // last cloud
                  if (int(laserCloudSurfLast->points[j].intensity) <
                      closestPointScan - 2.5) { // search another point within
                                                // 2.5 scans from pointSel
                    break;
                  }
                  // squared distance between the other point and pointSel
                  pointSqDis =
                      (laserCloudSurfLast->points[j].x - pointSel.x) *
                          (laserCloudSurfLast->points[j].x - pointSel.x) +
                      (laserCloudSurfLast->points[j].y - pointSel.y) *
                          (laserCloudSurfLast->points[j].y - pointSel.y) +
                      (laserCloudSurfLast->points[j].z - pointSel.z) *
                          (laserCloudSurfLast->points[j].z - pointSel.z);

                  if (int(laserCloudSurfLast->points[j].intensity) >=
                      closestPointScan) { // within same scan
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  } else { // not within same scan
                    if (pointSqDis < minPointSqDis3) {
                      minPointSqDis3 = pointSqDis;
                      minPointInd3 = j;
                    }
                  }
                }
              }

              // the planar patch
              //kd-tree 最近距离点，-1 表示没找到
              pointSearchSurfInd1[i] = closestPointInd;
			  //同一线号上的距离最近的点，-1 表示没找到满足的点
              pointSearchSurfInd2[i] = minPointInd2;
			  //不同线号上的距离最近的点，-1 表示没有找到满足的点
              pointSearchSurfInd3[i] = minPointInd3;
            }

			//找到了3 个点
            if (pointSearchSurfInd2[i] >= 0 &&
                pointSearchSurfInd3[i] >= 0) { // found the planar patch

				
                //A 点
              tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]];
				//B 点
              tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]];
				//C 点
              tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]];


				//向量AB X AC, 得到上一帧ABC 平面的法向量

				//x 轴方向的分量i
              float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z) -
                         (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
				//y 轴方向的分量j
              float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x) -
                         (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
				//z 轴方向的分量k
              float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y) -
                         (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
				//A点坐标与法向量的点乘，即A 点在法向量方向的投影
				//这里是把OA 在ABC 法向量上的投影分成了A 在法向量上的投影  
				//+ O 在法向量的投影
              float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

				//法向量的模
              float ps = sqrt(pa * pa + pb * pb + pc * pc);
				//pa pb pc  为法向量各方向上的单位向量
              pa /= ps;
              pb /= ps;
              pc /= ps;
              pd /= ps;

              // this is exactly the distance from pointSel to planar patch
              // {tripod1, tripod2, tripod3}
              //点到面的距离: 向量OA 在ABC 法向量上的投影，OA点乘n
              //pd 是A 在法向量上的投影，前面表示O 在法向量上的投影
              float pd2 =  //pa * (pointSel.x - tripod1.x) + pb * (pointSel.y - tripod1.y) + pc * (pointSel.z -tripod1.z)
                  pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

              // now
              // pa == diff(pd2, x0)
              // pb == diff(pd2, y0)
              // pc == diff(pd2, z0)

              /*pointProj = pointSel;
              pointProj.x -= pa * pd2;
              pointProj.y -= pb * pd2;
              pointProj.z -= pc * pd2;*/

			//计算权重
              float s = 1; // TODO: step? weight?
              if (iterCount >= 5) {
                s = 1 -
                    1.8 * fabs(pd2) /
                        sqrt(
                            sqrt(pointSel.x *
                                     pointSel.x // TODO: why adjust s like this?
                                 + pointSel.y * pointSel.y +
                                 pointSel.z * pointSel.z));
              }

			//考虑权重
              coeff.x = s * pa;
              coeff.y = s * pb;
              coeff.z = s * pc;
              coeff.intensity = s * pd2;

              if (s > 0.1 && pd2 != 0) { // apply this correspondence only when
                                         // s is not too small and distance is
                                         // not zero
                 //保存原始点和相应的系数
                laserCloudOri->push_back(surfPointsFlat->points[i]);
                coeffSel->push_back(coeff);
              }
            }
          }


		  //把对应好的特征点用最小二乘法求出最佳的对应关系，也就是里程计

          int pointSelNum = laserCloudOri->points.size();
		  //满足要求的特征点至少10 个，特征匹配数量太少弃用此帧数据
          if (pointSelNum < 10) {
            continue;
          }

          cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
          cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
          cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
          cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
          cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
          cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

		  //计算matA , matB 矩阵
          for (int i = 0; i < pointSelNum; i++) {
            pointOri = laserCloudOri->points[i]; // the original coordinates
                                                 // (not transformed to start
                                                 // time point) of feature point
                                                 // in current cloud
            coeff = coeffSel->points[i]; // the scaled gradients: diff(distance,
                                         // {x, y, z}), and the scaled distance.
                                         // x/y/z are coordinates of feature
                                         // points in the starting frame

            float s = 1;

			//计算点到世界坐标系的cos, sin 值
            float srx = sin(s * transform[0]);   // x -> pitch
            float crx = cos(s * transform[0]);
            float sry = sin(s * transform[1]);   //y->yaw
            float cry = cos(s * transform[1]);
            float srz = sin(s * transform[2]);	 //z->roll
            float crz = cos(s * transform[2]);
			//平移量
            float tx = s * transform[3];
            float ty = s * transform[4];
            float tz = s * transform[5];

			
			//error对分别对R 的三个分量求偏导，也即对R 求偏导

			//error 对rx 求偏导
            float arx =
                (-s * crx * sry * srz * pointOri.x +
                 s * crx * crz * sry * pointOri.y + s * srx * sry * pointOri.z +
                 s * tx * crx * sry * srz - s * ty * crx * crz * sry -
                 s * tz * srx * sry) *
                    coeff.x +
                (s * srx * srz * pointOri.x - s * crz * srx * pointOri.y +
                 s * crx * pointOri.z + s * ty * crz * srx - s * tz * crx -
                 s * tx * srx * srz) *
                    coeff.y +
                (s * crx * cry * srz * pointOri.x -
                 s * crx * cry * crz * pointOri.y - s * cry * srx * pointOri.z +
                 s * tz * cry * srx + s * ty * crx * cry * crz -
                 s * tx * crx * cry * srz) *
                    coeff.z;

			//error 对ry 求偏导
            float ary = ((-s * crz * sry - s * cry * srx * srz) * pointOri.x +
                         (s * cry * crz * srx - s * sry * srz) * pointOri.y -
                         s * crx * cry * pointOri.z +
                         tx * (s * crz * sry + s * cry * srx * srz) +
                         ty * (s * sry * srz - s * cry * crz * srx) +
                         s * tz * crx * cry) *
                            coeff.x +
                        ((s * cry * crz - s * srx * sry * srz) * pointOri.x +
                         (s * cry * srz + s * crz * srx * sry) * pointOri.y -
                         s * crx * sry * pointOri.z + s * tz * crx * sry -
                         ty * (s * cry * srz + s * crz * srx * sry) -
                         tx * (s * cry * crz - s * srx * sry * srz)) *
                            coeff.z;

			//error 对rz 求偏导
            float arz =
                ((-s * cry * srz - s * crz * srx * sry) * pointOri.x +
                 (s * cry * crz - s * srx * sry * srz) * pointOri.y +
                 tx * (s * cry * srz + s * crz * srx * sry) -
                 ty * (s * cry * crz - s * srx * sry * srz)) *
                    coeff.x +
                (-s * crx * crz * pointOri.x - s * crx * srz * pointOri.y +
                 s * ty * crx * srz + s * tx * crx * crz) *
                    coeff.y +
                ((s * cry * crz * srx - s * sry * srz) * pointOri.x +
                 (s * crz * sry + s * cry * srx * srz) * pointOri.y +
                 tx * (s * sry * srz - s * cry * crz * srx) -
                 ty * (s * crz * sry + s * cry * srx * srz)) *
                    coeff.z;


			//error 对t 求偏导
            float atx = -s * (cry * crz - srx * sry * srz) * coeff.x +
                        s * crx * srz * coeff.y -
                        s * (crz * sry + cry * srx * srz) * coeff.z;

            float aty = -s * (cry * srz + crz * srx * sry) * coeff.x -
                        s * crx * crz * coeff.y -
                        s * (sry * srz - cry * crz * srx) * coeff.z;

            float atz = s * crx * sry * coeff.x - s * srx * coeff.y -
                        s * crx * cry * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = atx;
            matA.at<float>(i, 4) = aty;
            matA.at<float>(i, 5) = atz;
			//这里为什么要乘以0.05 ? 低通滤波?
            matB.at<float>(i, 0) = -0.05 * d2;

			
          }

		  
          cv::transpose(matA, matAt);
          matAtA = matAt * matA;
          matAtB = matAt * matB;
		  //求解matAtA * matX = matAtB
		  //x = (matAtA)^-1 * matAtB
          cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);








		  

          if (iterCount == 0) { // initialize
          //特征值1*6矩阵
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
		  //特征向量6*6矩阵
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

			//求解特征值，特征向量
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
			//特征值取值门槛
            float eignThre[6] = {10, 10, 10, 10, 10, 10};
			//从小到大查找
            for (int i = 5; i >= 0; i--) {
				//特征值太小，则认为处在兼并环境中，发生了退化
              if (matE.at<float>(0, i) < eignThre[i]) {
			  	//对应的特征向量置为0
                for (int j = 0; j < 6; j++) {
                  matV2.at<float>(i, j) = 0;
                }
                isDegenerate = true;
              } else {
                break;
              }
            }
			//计算特征向量的协方差矩阵
            matP = matV.inv() * matV2;
          }

			//如果发生退化
          if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
			//通过特征向量的协方差更新X 矩阵
            matX = matP * matX2;
          }

			//累加每次迭代的旋转平移量
          transform[0] += matX.at<float>(0, 0);
          transform[1] += matX.at<float>(1, 0);
          transform[2] += matX.at<float>(2, 0);
          transform[3] += matX.at<float>(3, 0);
          transform[4] += matX.at<float>(4, 0);
          transform[5] += matX.at<float>(5, 0);

          for (int i = 0; i < 6; i++) {
		  	//判断是否非数字
            if (std::isnan(transform[i]))
              transform[i] = 0;
          }
		  //计算旋转平移量，如果很小就停止迭代
          float deltaR = sqrt(pow(rad2deg(matX.at<float>(0, 0)), 2) +
                              pow(rad2deg(matX.at<float>(1, 0)), 2) +
                              pow(rad2deg(matX.at<float>(2, 0)), 2));
          float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
                              pow(matX.at<float>(4, 0) * 100, 2) +
                              pow(matX.at<float>(5, 0) * 100, 2));

		//迭代终止条件
          if (deltaR < 0.1 && deltaT < 0.1) {
            break;
          }
        }
      }

      // accumulate transform
      float rx, ry, rz, tx, ty, tz;
	  //求相对于原点的旋转量，垂直方向上1.05 倍的修正
	  //transformSum 是上一帧相对于第一帧的状态转移矩阵
	  //transform上当前帧相对于上一帧的状态转移矩阵
	  //现在需要计算当前帧相对于第一帧的状态转移矩阵
	  //可以通过状态转移矩阵中的姿态，求出对应的旋转矩阵R
	  //RtransformSum(n) =RtransformSum(n-1)^T * Rtransform(n)
	  //再通过临时的RtransformSum(n) 求出对应的pitch, roll,yaw, 也就是状态转移矩阵的旋转量
      AccumulateRotation(transformSum[0], transformSum[1], transformSum[2],
                         -transform[0], -transform[1] * 1.05, -transform[2], rx,
                         ry, rz);

		//根据求出来的当前帧相对于第一帧的里程计姿态把
		//平移量旋转过去

		//绕Z 轴旋转，这里先把点云位置从开始位置平移到结束位置
      float x1 = cos(rz) * (transform[3] - imuShiftFromStartX) -
                 sin(rz) * (transform[4] - imuShiftFromStartY);
      float y1 = sin(rz) * (transform[3] - imuShiftFromStartX) +
                 cos(rz) * (transform[4] - imuShiftFromStartY);
      float z1 = transform[5] * 1.05 - imuShiftFromStartZ;


		//绕x 轴旋转
      float x2 = x1;
      float y2 = cos(rx) * y1 - sin(rx) * z1;
      float z2 = sin(rx) * y1 + cos(rx) * z1;

		//绕y 轴旋转
      tx = transformSum[3] - (cos(ry) * x2 + sin(ry) * z2);
      ty = transformSum[4] - y2;
      tz = transformSum[5] - (-sin(ry) * x2 + cos(ry) * z2);

	//利用当前帧的开始点云imu姿态和结束点云imu姿态对里程计进行修正

	
      PluginIMURotation(rx, ry, rz, imuPitchStart, imuYawStart, imuRollStart,
                        imuPitchLast, imuYawLast, imuRollLast, rx, ry, rz);

	//最后得到当前帧结束点云到第一帧开始点云的里程计
      transformSum[0] = rx;
      transformSum[1] = ry;
      transformSum[2] = rz;
      transformSum[3] = tx;
      transformSum[4] = ty;
      transformSum[5] = tz;

	//欧拉角转换成四元素
      geometry_msgs::Quaternion geoQuat =
          tf::createQuaternionMsgFromRollPitchYaw(rz, -rx, -ry);

		//publish 四元素和平移量
      laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
      laserOdometry.pose.pose.orientation.x = -geoQuat.y;
      laserOdometry.pose.pose.orientation.y = -geoQuat.z;
      laserOdometry.pose.pose.orientation.z = geoQuat.x;
      laserOdometry.pose.pose.orientation.w = geoQuat.w;
      laserOdometry.pose.pose.position.x = tx;
      laserOdometry.pose.pose.position.y = ty;
      laserOdometry.pose.pose.position.z = tz;
      pubLaserOdometry.publish(laserOdometry);

      // laserOdometryFreq.report();

		//广播新的平移旋转之后的坐标系(rviz)
      laserOdometryTrans.stamp_ = ros::Time().fromSec(timeSurfPointsLessFlat);
      laserOdometryTrans.setRotation(
          tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
      laserOdometryTrans.setOrigin(tf::Vector3(tx, ty, tz));
      tfBroadcaster.sendTransform(laserOdometryTrans);

	//对点云的角点和平面点投影到扫描结束位置，
      int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
      for (int i = 0; i < cornerPointsLessSharpNum; i++) {
        TransformToEnd(&cornerPointsLessSharp->points[i],
                       &cornerPointsLessSharp->points[i]);
      }

      int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
      for (int i = 0; i < surfPointsLessFlatNum; i++) {
        TransformToEnd(&surfPointsLessFlat->points[i],
                       &surfPointsLessFlat->points[i]);
      }

      frameCount++;
	  //点云全部点，每隔一个点云数据相对点云最后一个点
	  //进行畸变校正
      if (frameCount >= skipFrameNum + 1) {
        int laserCloudFullResNum = laserCloudFullRes->points.size();
        for (int i = 0; i < laserCloudFullResNum; i++) {
          TransformToEnd(
              &laserCloudFullRes->points[i],
              &laserCloudFullRes
                   ->points[i]); // transform all points in this sweep to end
        }
      }

		//畸变校正之后的点作为last 点保存等下一个点云来进行匹配
      pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
      cornerPointsLessSharp = laserCloudCornerLast;
      laserCloudCornerLast = laserCloudTemp;

      laserCloudTemp = surfPointsLessFlat;
      surfPointsLessFlat = laserCloudSurfLast;
      laserCloudSurfLast = laserCloudTemp;

      laserCloudCornerLastNum = laserCloudCornerLast->points.size();
      laserCloudSurfLastNum = laserCloudSurfLast->points.size();

	  //点足够多就构建kd-tree, 否则弃用次帧，沿用上一帧
	  //数据的kd-tree
      if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {
        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
      }
//按照跳帧数publish 边缘点，平面点以及全部点给laserMapping
//(每隔一帧发一次)
      if (frameCount >= skipFrameNum + 1) {
        frameCount = 0;

        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp =
            ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(
            laserCloudCornerLast2); // all transformed to sweep end

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp =
            ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(
            laserCloudSurfLast2); // all transformed to sweep end

        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp =
            ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudFullRes3.header.frame_id = "/camera";
        pubLaserCloudFullRes.publish(
            laserCloudFullRes3); // all transformed to sweep end

        // registeredLaserCloudFreq.report();
      }
    }

    status = ros::ok();
    rate.sleep();
  }

  return 0;
}
