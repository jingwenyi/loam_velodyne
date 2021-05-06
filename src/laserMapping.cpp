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
//���ƽ��յ��ĵ������ݣ�ÿ����֡����һ��
const int stackFrameNum = 1;
//���ƴ���õ��ĵ���map, ÿ������publish ��rviz ��ʾ
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

//�����ǵ�ͼ����(center) �ڵ�ͼ�е�λ�õĳ�ʼ��ֵ
//���߰ѵ�ͼԭ���ʼ��ֵ���������ɽ�ͼ��Χ������
int laserCloudCenWidth = 10;
int laserCloudCenHeight = 5;
int laserCloudCenDepth = 10;



//��ͼ���������cube ��������ʵ�ĵ�ͼ��С��50����
//�Ӷ����Լ����������ͼ��width ���� �ܹ���ʾ�ľ�����50*21 = 1050 m
//��height �ķ������ܹ���ʾ�ľ�����11 * 50 = 550m
//��depth �ķ������ܹ���ʾ�ľ�����21 * 50 = 1050m
//��cube  �ؿ���ķָ����
const int laserCloudWidth = 21;
//��cube �ظ߷���ķָ����
const int laserCloudHeight = 11;
//�� cube ����ȷ���ָ����
const int laserCloudDepth = 21;


//��ͼ�Ĵ�С, �����Ϊ1050 * 550 * 1050 = 606375000 m*m*m
//��cube ����4851
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;

//lidar ����Χ��(FOV) �ĵ��Ƽ�����
int laserCloudValidInd[125];
//lidar ��Χ���Ƽ�����
int laserCloudSurroundInd[125];


//���½��յ���������(��Ե + ƽ��)

pcl::PointCloud<PointType>::Ptr
    laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    laserCloudSurfLast(new pcl::PointCloud<PointType>());


//��ŵ�ǰ�յ���������(�²���֮���)
pcl::PointCloud<PointType>::Ptr
    laserCloudCornerStack(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    laserCloudSurfStack(new pcl::PointCloud<PointType>());

//��ŵ�ǰ�յ���������(�²���֮ǰ)
pcl::PointCloud<PointType>::Ptr
    laserCloudCornerStack2(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    laserCloudSurfStack2(new pcl::PointCloud<PointType>());


//ԭʼ���������Ȩ��
pcl::PointCloud<PointType>::Ptr laserCloudOri(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr coeffSel(new pcl::PointCloud<PointType>());


//ƥ��ʹ�õ�������(�²���֮��)
pcl::PointCloud<PointType>::Ptr
    laserCloudSurround(new pcl::PointCloud<PointType>());
//ƥ��ʹ�õ�������(�²���֮ǰ)
pcl::PointCloud<PointType>::Ptr
    laserCloudSurround2(new pcl::PointCloud<PointType>());



//map ����ȡ��ƥ��ʹ�õ�������
pcl::PointCloud<PointType>::Ptr
    laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr
    laserCloudSurfFromMap(new pcl::PointCloud<PointType>());


//����ȫ����
pcl::PointCloud<PointType>::Ptr
    laserCloudFullRes(new pcl::PointCloud<PointType>());


//array ������50 m Ϊ��λ���������ͼ�����й����л�һֱ����
//(����Ҫ�Ļ����Կ����Ż���ֻ�����ڽ���)

//����������������cube
//�����ǰ�һ����ά���鱣����һ��һά������ [21][11][21]
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

//�м����������²�����������

pcl::PointCloud<PointType>::Ptr laserCloudCornerArray2[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray2[laserCloudNum];



//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr
    kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr
    kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

// rotation {x, y, z} or {pitch?, yaw?, roll?}, translation {x, y, z}
//��ǰ֡�����㵽��һ֡��ʼ���״̬ת�ƾ���Ҳ������̼�



//odometry ����õ��ĵ���������ϵ�µ�ת�ƾ���T
float transformSum[6] = {0};

//ת��������ֻʹ���˺�����ƽ������
float transformIncre[6] = {0};


//����ʼλ��Ϊԭ�����������ϵ�µ�ת������(�²�������Ķ���)
float transformTobeMapped[6] = {0};

//���mapping ֮ǰ��Odometry �������������ϵ��ת�ƾ���(��Ƶ������һ����transformSum һ��)
float transformBefMapped[6] = {0};

//���mapping ֮��ľ���mapping ΢��֮���ת������
float transformAftMapped[6] = {0};

int imuPointerFront = 0;
int imuPointerLast = -1;
const int imuQueLength = 200;

double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};

//��������ģ�ͣ������ϴ�΢���Ľ����odometry ������ϴμ���Ľ��
//�²�һ���µ���������ϵ��ת������transformTobeMapped


void transformAssociateToMap() {
	//����һ�ν�ͼʱʹ�õĽ�ͼ֡������������� 
	//��һ֡��ʼ���Ƶ���̼�״̬����
	//��ת����ǰ֡�������ƵĻ�������ϵ��

	//  ��ƽ�ƣ��� ��y �� (-yaw) ��ת
  float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) -
             sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
  float y1 = transformBefMapped[4] - transformSum[4];
  float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) +
             cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
	//��x ��(-pitch) ��ת
  float x2 = x1;
  float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
  float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

	//��z ��(-roll) ��ת
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

	//ͨ����һ�ν�ͼʱʹ�õ� ��ת����(����̼�)�� 
	//�͵�ǰ֡����ת���������ǰ֡
	//��Ӧ�ĵ�ͼ��ת���󣬴Ӷ����λ��ת�ƾ���
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

  //������ĵ�ǰ֡��Ӧ�ĵ�ͼ��̬���󣬴ӵ�ǰ֡�Ļ�������ת����������

	//��z ����ת(roll)
  x1 = cos(transformTobeMapped[2]) * transformIncre[3] -
       sin(transformTobeMapped[2]) * transformIncre[4];
  y1 = sin(transformTobeMapped[2]) * transformIncre[3] +
       cos(transformTobeMapped[2]) * transformIncre[4];
  z1 = transformIncre[5];

	//��x ����ת(pitch)
  x2 = x1;
  y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
  z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

	//�� y ����ת(yaw), ��ƽ��
  transformTobeMapped[3] =
      transformAftMapped[3] -
      (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
  transformTobeMapped[4] = transformAftMapped[4] - y2;
  transformTobeMapped[5] =
      transformAftMapped[5] -
      (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
}

//��¼odometry ���͵�ת��������mapping֮���ת������
void transformUpdate() {
  if (imuPointerLast >= 0) {
    float imuRollLast = 0, imuPitchLast = 0;
	//���ҵ���ʱ���С��imu ʱ�����imu λ��
    while (imuPointerFront != imuPointerLast) {
      if (timeLaserOdometry + scanPeriod < imuTime[imuPointerFront]) {
        break;
      }
      imuPointerFront = (imuPointerFront + 1) % imuQueLength;
    }

    if (timeLaserOdometry + scanPeriod > imuTime[imuPointerFront]) {
		//δ�ҵ�����ʱimuPointerFront == imuPointerLast
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
		//��ʱ������󷭹��Ǻ͸�����
      imuRollLast = imuRoll[imuPointerFront] * ratioFront +
                    imuRoll[imuPointerBack] * ratioBack;
      imuPitchLast = imuPitch[imuPointerFront] * ratioFront +
                     imuPitch[imuPointerBack] * ratioBack;
    }

	//��imu ���ݽ��е�ͨ�˲�
    transformTobeMapped[0] =
        0.998 * transformTobeMapped[0] + 0.002 * imuPitchLast;
    transformTobeMapped[2] =
        0.998 * transformTobeMapped[2] + 0.002 * imuRollLast;
  }

	//��¼�Ż�֮ǰ���Ż�֮���ת�ƾ���
  for (int i = 0; i < 6; i++) {
    transformBefMapped[i] = transformSum[i];
    transformAftMapped[i] = transformTobeMapped[i];
  }
}


//���ݵ���������ת�ƾ��󣬽���ע�ᵽȫ����������ϵ��
void pointAssociateToMap(PointType const *const pi, PointType *const po) {
	//��z ��(roll) ��ת(transformToMapped[2])
  float x1 =
      cos(transformTobeMapped[2]) * pi->x - sin(transformTobeMapped[2]) * pi->y;
  float y1 =
      sin(transformTobeMapped[2]) * pi->x + cos(transformTobeMapped[2]) * pi->y;
  float z1 = pi->z;

	//��x ��(pitch) ��ת(transformToMapped[0])
  float x2 = x1;
  float y2 =
      cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
  float z2 =
      sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

	//��y ��(yaw) ��ת(transformToMapped[1])����ƽ��
  po->x = cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2 +
          transformTobeMapped[3];
  po->y = y2 + transformTobeMapped[4];
  po->z = -sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2 +
          transformTobeMapped[5];
  po->intensity = pi->intensity;
}

//��ת�Ƶ��ֲ�����ϵ��
void pointAssociateTobeMapped(PointType const *const pi, PointType *const po) {
//ƽ�ƺ���y ����ת(-transformTobeMapped[1])
  float x1 = cos(transformTobeMapped[1]) * (pi->x - transformTobeMapped[3]) -
             sin(transformTobeMapped[1]) * (pi->z - transformTobeMapped[5]);
  float y1 = pi->y - transformTobeMapped[4];
  float z1 = sin(transformTobeMapped[1]) * (pi->x - transformTobeMapped[3]) +
             cos(transformTobeMapped[1]) * (pi->z - transformTobeMapped[5]);
//��x ����ת(-transformTobeMapped[0])
  float x2 = x1;
  float y2 =
      cos(transformTobeMapped[0]) * y1 + sin(transformTobeMapped[0]) * z1;
  float z2 =
      -sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;
//��z ����ת(-transformTobeMapp[2])
  po->x = cos(transformTobeMapped[2]) * x2 + sin(transformTobeMapped[2]) * y2;
  po->y = -sin(transformTobeMapped[2]) * x2 + cos(transformTobeMapped[2]) * y2;
  po->z = z2;
  po->intensity = pi->intensity;
}

//���ձ��ص�
void laserCloudCornerLastHandler(
    const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2) {
  timeLaserCloudCornerLast = laserCloudCornerLast2->header.stamp.toSec();

  laserCloudCornerLast->clear();
  pcl::fromROSMsg(*laserCloudCornerLast2, *laserCloudCornerLast);

  newLaserCloudCornerLast = true;
}


//����ƽ���
void laserCloudSurfLastHandler(
    const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2) {
  timeLaserCloudSurfLast = laserCloudSurfLast2->header.stamp.toSec();

  laserCloudSurfLast->clear();
  pcl::fromROSMsg(*laserCloudSurfLast2, *laserCloudSurfLast);

  newLaserCloudSurfLast = true;
}

//���յ���ȫ����
void laserCloudFullResHandler(
    const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2) {
  timeLaserCloudFullRes = laserCloudFullRes2->header.stamp.toSec();

  laserCloudFullRes->clear();
  pcl::fromROSMsg(*laserCloudFullRes2, *laserCloudFullRes);

  newLaserCloudFullRes = true;
}

//������תƽ����Ϣ
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry) {
  timeLaserOdometry = laserOdometry->header.stamp.toSec();

  double roll, pitch, yaw;
  //��Ԫ��ת��Ϊŷ����
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

//����IMU ��Ϣ��ֻʹ���˷����Ǻ͸�����
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

	//��������դ���˲���
  pcl::VoxelGrid<PointType> downSizeFilterCorner;
	//�������ش�С
  downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);

  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);

  pcl::VoxelGrid<PointType> downSizeFilterMap;
  downSizeFilterMap.setLeafSize(0.6, 0.6, 0.6);

	//ָ���ʼ��
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
	  	//ͨ����һ֡�ĵ�ͼλ�� ����͵�ǰ֡����̼���ǰ֡�ĵ�ͼλ�˾���
        transformAssociateToMap();


		//�ѵ�ǰ���ص���ת�����µ�ͼ����ϵ��
        int laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        for (int i = 0; i < laserCloudCornerLastNum; i++) {
			//�ѵ���ת���µ���������ϵ��
          pointAssociateToMap(&laserCloudCornerLast->points[i], &pointSel);
		  //����ת�����µ�ͼ����ϵ�еı��������㣬���浽laserCloudCornerStack2 ��
          laserCloudCornerStack2->push_back(pointSel);
        }

		//�ѵ�ǰƽ�����ת�����µ�ͼ����ϵ��
        int laserCloudSurfLastNum = laserCloudSurfLast->points.size();
        for (int i = 0; i < laserCloudSurfLastNum; i++) {
			//�ѵ���������µĵ�ͼ����ϵ��
          pointAssociateToMap(&laserCloudSurfLast->points[i], &pointSel);
			//�ѹ�������������ϵ��ƽ�������㣬���浽laserCloudSurfStack2 ��
          laserCloudSurfStack2->push_back(pointSel);
        }
      }

      if (frameCount >= stackFrameNum) {
        frameCount = 0;

        PointType pointOnYAxis;
        pointOnYAxis.x = 0.0;
        pointOnYAxis.y = 10.0;
        pointOnYAxis.z = 0.0;
		//��ȡy ������10 m ��λ�õĵ�����������ϵ�µ�����
        pointAssociateToMap(&pointOnYAxis, &pointOnYAxis);

		//�������е�����������ϵ�µ�(ԭ��) λ��
		//����ȡһ(��50 m �������������Ч��)�����������±�
		//Ϊ����������ͼ���ܽ�����ԭ��ǰ�����
		//ÿһάƫ��һ��laserCloudCenWidth (��ֵ�ᶯ̬��������
		//ʹ������������󻯣���ʼֵΪ�����鳤��1/2) ����



		//ͨ��transformAssociateToMap �²���µ������������ת�����е�ƽ����
		//����õ���֡�����ĵ��ڵ�ͼ�е��Ǹ���������
        int centerCubeI =
            int((transformTobeMapped[3] + 25.0) / 50.0) + laserCloudCenWidth;
        int centerCubeJ =
            int((transformTobeMapped[4] + 25.0) / 50.0) + laserCloudCenHeight;
        int centerCubeK =
            int((transformTobeMapped[5] + 25.0) / 50.0) + laserCloudCenDepth;

		//���ڼ��������������ȡ����Ϊ�˲�ʹ(-50.0,50.0) �����
		//������ƫ�ƣ�����������Ϊ����ʱ������ͳһ����
		//һ����λ��Ҳ����һ
        if (transformTobeMapped[3] + 25.0 < 0)
          centerCubeI--;
        if (transformTobeMapped[4] + 25.0 < 0)
          centerCubeJ--;
        if (transformTobeMapped[5] + 25.0 < 0)
          centerCubeK--;



		//����������ͼ�ڸ���ά�ռ�(��ʵ��һ��һά����) �е�λ�ã���ֹ��ͼ��ǰɨ�赽������
		//Խ�磬���ո�˼·��������Ϊ���һֱ��һ�����潨ͼ�������Ǹ������ܹ����ֵ�
		//����󣬾ͻᶪ��֮ǰ�ĵ�ͼ����


		//���ڼ����״�ɨ�跶Χ��150m �ڣ��ʵ�ͼԭ��������߽磬������3 ��������
		//����֮��ȡֵ��Χ: 3< centerCubeI < 18, 3 < centerCubeJ < 8,
		// 3<centerCubeK < 18

		//��������±߽磬������ͼ�򸺷�������Ŀ����ԱȽϴ�,
		//��ѭ����λ�����������ĵ����ϱ߽����һ����λ

        // shift block of point cloud

        while (centerCubeI < 3) {
          for (int j = 0; j < laserCloudHeight; j++) {
            for (int k = 0; k < laserCloudDepth; k++) {
				//ʵ��һ��ѭ����λ��Ч��
              int i = laserCloudWidth - 1;
				//ָ�븳ֵ���������һ��ָ��λ��
				//�����������һ����ά���鱣����һ��һά��������
				//�������һ����������i + 
              pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                  laserCloudCornerArray[i + laserCloudWidth * j +
                                        laserCloudWidth * laserCloudHeight * k];
              pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                  laserCloudSurfArray[i + laserCloudWidth * j +
                                      laserCloudWidth * laserCloudHeight * k];
				//ѭ����λ��I γ�������κ���
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

			  //����ʼ�㸳ֵΪ���һ����
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


		//��������ϱ߽磬������ͼ������������Ŀ����ԱȽϴ�
		//��ѭ����λ�����������ĵ����±߽����һ����λ
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
				//I γ��������ǰ��
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
				//J γ���ϣ����κ���
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
				//J ά��������ǰ��
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
				//K ά�������κ���
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
				//kά��������ǰ��
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







//���ϵ�ͼ�������





//��ȡ��ǰ֡��Ӧ��ͼ�����ĵ�ͼ�㣬���ڼ����״��ɨ�����Ϊ0-150m
//������ÿ��cub �ľ�����50m , ����ǰ������ֻ��Ҫ


		

        int laserCloudValidNum = 0;
        int laserCloudSurroundNum = 0;

		

		//��ÿһά����5 ��cube (ǰ2������2 �����м�1��)����в���
		//(ǰ��250m��Χ�ڣ��ܹ�500m ��Χ)������ά���ܹ�125��cube





		
		//����125 ��cube �����һ��ɸѡ������Χ�ڵ�cube		
        for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++) {
          for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) {
            for (int k = centerCubeK - 2; k <= centerCubeK + 2; k++) {
              if (i >= 0 && i < laserCloudWidth && j >= 0 &&
                  j < laserCloudHeight && k >= 0 && k < laserCloudDepth) {
                  //��������Ϸ�

				  
					//�����ʵ�ʱ���������������ϵ�µ�����
                float centerX = 50.0 * (i - laserCloudCenWidth);
                float centerY = 50.0 * (j - laserCloudCenHeight);
                float centerZ = 50.0 * (k - laserCloudCenDepth);

				//�ж��Ƿ�lidar ���߷�Χ�ı�־
                bool isInLaserFOV = false;
                for (int ii = -1; ii <= 1; ii += 2) {
                  for (int jj = -1; jj <= 1; jj += 2) {
                    for (int kk = -1; kk <= 1; kk += 2) {
						//��������8 ����������
                      float cornerX = centerX + 25.0 * ii;
                      float cornerY = centerY + 25.0 * jj;
                      float cornerZ = centerZ + 25.0 * kk;
						//ԭ�㵽��������ƽ����
                      float squaredSide1 =
                          (transformTobeMapped[3] - cornerX) *
                              (transformTobeMapped[3] - cornerX) +
                          (transformTobeMapped[4] - cornerY) *
                              (transformTobeMapped[4] - cornerY) +
                          (transformTobeMapped[5] - cornerZ) *
                              (transformTobeMapped[5] - cornerZ);

						//fov �趨�ķ�ΧΪ60��,�ϱ�30�ȣ��±�30�ȣ�cos 30 = sqrt(3.0)/2
						//���Ҷ���cosA = (b*b + c*c - a*a) / (2*b*c)

						//pointOnYAxis ����������ƽ����
					  float squaredSide2 = (pointOnYAxis.x - cornerX) *
                                               (pointOnYAxis.x - cornerX) +
                                           (pointOnYAxis.y - cornerY) *
                                               (pointOnYAxis.y - cornerY) +
                                           (pointOnYAxis.z - cornerZ) *
                                               (pointOnYAxis.z - cornerZ);

					��
						//����ԭ�㵽����ľ������������Բ�������check1, check2
						//|OP|^2 + |OD|^2 - |PD|^2  < 2*|OP||OD|*cos(30)        (OD Ϊ������squaredSide1  > 0)
                      float check1 = 100.0 + squaredSide1 - squaredSide2 -
                                     10.0 * sqrt(3.0) * sqrt(squaredSide1);

						//|OP|^2 + |OD|^2 - |PD|^2  <  -2*|OP||OD|*cos(30)       (OD Ϊ������squaredSide1  < 0)
                      float check2 = 100.0 + squaredSide1 - squaredSide2 +
                                     10.0 * sqrt(3.0) * sqrt(squaredSide1);

                      if (check1 < 0 && check2 > 0) {
                        isInLaserFOV = true;
                      }
                    }
                  }
                }

				//��ס����Χ�ڵ�cube ������ƥ����
                if (isInLaserFOV) {
                  laserCloudValidInd[laserCloudValidNum] =
                      i + laserCloudWidth * j +
                      laserCloudWidth * laserCloudHeight * k;
                  laserCloudValidNum++;
                }
				//��ס��������cube ����������ʾ��
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
		//�����������ͼ������ƥ��ʹ��
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
		//�����˲�����
        downSizeFilterCorner.setInputCloud(laserCloudCornerStack2);
		//ִ���˲�����
        downSizeFilterCorner.filter(*laserCloudCornerStack);
		//��ȡ�˲������ص�ߴ�
        int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

        laserCloudSurfStack->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfStack2);
        downSizeFilterSurf.filter(*laserCloudSurfStack);
        int laserCloudSurfStackNum = laserCloudSurfStack->points.size();

        laserCloudCornerStack2->clear();
        laserCloudSurfStack2->clear();





		//�õ�ͼ����Ұ��Χ�ڵĵ��ƽ���kd-tree (����������ƽ������)
		//�ٶԵ�ǰ֡���ƺ͵�ͼ����Ұ�ڵĵ��ƽ�����׼(ͬ��̼�)
		//�������С���˷�������ת�ƾ���

        if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 100) {
			//�������ص��kd-tree
          kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
			//����ƽ����kd-tree
          kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);

			//��ʼ����
          for (int iterCount = 0; iterCount < maxIterNum; iterCount++) {
            laserCloudOri->clear();
            coeffSel->clear();

            // find correspondance
            //�����������ƥ��
            for (int i = 0; i < laserCloudCornerStackNum; i++) {
              pointOri = laserCloudCornerStack->points[i];
			  //��ȡ�õ��ڵ�ͼ�е�����
              pointAssociateToMap(&pointOri, &pointSel);
			  //Ѱ��������������5 ����
              kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                                  pointSearchSqDis);

				//5 ���������ľ��벻����1 �Ŵ���
              if (pointSearchSqDis[4] < 1.0) {
			  	//���������������Ӻ���ƽ��
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

				//�������
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

				//��������
                matA1.at<float>(0, 0) = a11;
                matA1.at<float>(0, 1) = a12;
                matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12;
                matA1.at<float>(1, 1) = a22;
                matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13;
                matA1.at<float>(2, 1) = a23;
                matA1.at<float>(2, 2) = a33;

				//����ֵ�ֽ�

                cv::eigen(matA1, matD1, matV1);

				//�����������ֵ���ڵڶ��������ֵ��������
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


				  //Ȩ��ϵ��

                  float s = 1 - 0.9 * fabs(ld2);

                  coeff.x = s * la;
                  coeff.y = s * lb;
                  coeff.z = s * lc;
                  coeff.intensity = s * ld2;

					//�����㹻С��ʹ��
                  if (s > 0.1) {
                    laserCloudOri->push_back(pointOri);
                    coeffSel->push_back(coeff);
                  }
                }
              }
            }
            // find correspondance
            //ƽ���������ƥ��
            for (int i = 0; i < laserCloudSurfStackNum; i++) {
              pointOri = laserCloudSurfStack->points[i];
              pointAssociateToMap(&pointOri, &pointSel);
              kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                                pointSearchSqDis);

              if (pointSearchSqDis[4] < 1.0) {
			  	//��������������������
                for (int j = 0; j < 5; j++) {
                  matA0.at<float>(j, 0) =
                      laserCloudSurfFromMap->points[pointSearchInd[j]].x;
                  matA0.at<float>(j, 1) =
                      laserCloudSurfFromMap->points[pointSearchInd[j]].y;
                  matA0.at<float>(j, 2) =
                      laserCloudSurfFromMap->points[pointSearchInd[j]].z;
                }
				//��� matA0 * matX0 = matB0
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
			//���������̫��
            if (laserCloudSelNum < 50) {
              continue;
            }



			//��С���˷��������ת����

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

			//�˻������ж��봦��
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



			//ͨ��������ת�ƾ�������֮ǰ�²����������ϵ�µ���ת����
			//����ÿ�εĵ�����
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
			//��תƽ�����㹻С��ֹͣ����
            if (deltaR < 0.05 && deltaT < 0.05) {
              break;
            }
          }

			//�������������ص�ת�ƾ���
          transformUpdate();
        }




		//---------------------------��ͼ----------------


		

		//��corner points ������(��������С) ������Ӧ��������
        for (int i = 0; i < laserCloudCornerStackNum; i++) {
			//ת�Ƶ���������
          pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);

			//��50 �ı�������С���������룬ƫ��laserCloudCen* ��
			//������������
			//�����25.0, �����ڼ�һ��laserCloudCenWidth  ʵ�������������Ч��
          int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
          int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
          int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

			//���pointSel.x С��-25����������ķ���ͷ��ˣ���Ҫ��ȥ1
          if (pointSel.x + 25.0 < 0)
            cubeI--;
          if (pointSel.y + 25.0 < 0)
            cubeJ--;
          if (pointSel.z + 25.0 < 0)
            cubeK--;

          if (cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 &&
              cubeJ < laserCloudHeight && cubeK >= 0 &&
              cubeK < laserCloudDepth) {
              //ֻ��ѡ-laserCloudCenWidth*50.0 < point.x < laserCloudCenWidth * 50.0
              //��Χ�ڵĵ㣬y ��z ͬ���ճߴ�Ž���ͬ���飬
              //ÿ����ĵ���������


			  //����������������������id
            int cubeInd = cubeI + laserCloudWidth * cubeJ +
                          laserCloudWidth * laserCloudHeight * cubeK;
			  //����������뵽��������
            laserCloudCornerArray[cubeInd]->push_back(pointSel);
          }
        }


		

		//��surf points������(��������С)������Ӧ��������
        for (int i = 0; i < laserCloudSurfStackNum; i++) {
			//ת�Ƶ���������ϵ
          pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

			//����ƽ����������ڵ�������
          int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
          int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
          int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;


			////���pointSel.x С��-25����������ķ���ͷ��ˣ���Ҫ��ȥ1
          if (pointSel.x + 25.0 < 0)
            cubeI--;
          if (pointSel.y + 25.0 < 0)
            cubeJ--;
          if (pointSel.z + 25.0 < 0)
            cubeK--;


			//������������ͼ��
          if (cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 &&
              cubeJ < laserCloudHeight && cubeK >= 0 &&
              cubeK < laserCloudDepth) {
              //���������������������id
            int cubeInd = cubeI + laserCloudWidth * cubeJ +
                          laserCloudWidth * laserCloudHeight * cubeK;
			  //������������ͼ��Ӧ��������
            laserCloudSurfArray[cubeInd]->push_back(pointSel);
          }
        }


			//�������²���

		//�Ե�ͼ�е�ÿ��cub ����������²���
        for (int i = 0; i < laserCloudValidNum; i++) {
          int ind = laserCloudValidInd[i];

          laserCloudCornerArray2[ind]->clear();
          downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
          downSizeFilterCorner.filter(*laserCloudCornerArray2[ind]);

          laserCloudSurfArray2[ind]->clear();
          downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
          downSizeFilterSurf.filter(*laserCloudSurfArray2[ind]);
			//Array ��Array2 ���������˲������Ҹ���
          pcl::PointCloud<PointType>::Ptr laserCloudTemp =
              laserCloudCornerArray[ind];
          laserCloudCornerArray[ind] = laserCloudCornerArray2[ind];
          laserCloudCornerArray2[ind] = laserCloudTemp;

          laserCloudTemp = laserCloudSurfArray[ind];
          laserCloudSurfArray[ind] = laserCloudSurfArray2[ind];
          laserCloudSurfArray2[ind] = laserCloudTemp;
        }
		//��ͼ֡����1
        mapFrameCount++;




		
		//����������²�����ÿ����֡publish һ�Σ��ӵ�һ�ο�ʼ
        if (mapFrameCount >= mapFrameNum) {
          mapFrameCount = 0;

          laserCloudSurround2->clear();


			//�� ��ͼ��  �����������ƽ����������л���
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

			//�Ի��ܵ� ����������²�����
			
          laserCloudSurround->clear();
          downSizeFilterCorner.setInputCloud(laserCloudSurround2);
          downSizeFilterCorner.filter(*laserCloudSurround);

			//������ͼ�²�������ܵ�������
			//��������������ƽ������
          sensor_msgs::PointCloud2 laserCloudSurround3;
          pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
          laserCloudSurround3.header.stamp =
              ros::Time().fromSec(timeLaserOdometry);
          laserCloudSurround3.header.frame_id = "/camera_init";
          pubLaserCloudSurround.publish(laserCloudSurround3);

          // mappingCloudFreq.report();
        }


			//��������ȫ����ת�Ƶ���������ϵ��
        int laserCloudFullResNum = laserCloudFullRes->points.size();
        for (int i = 0; i < laserCloudFullResNum; i++) {
          pointAssociateToMap(&laserCloudFullRes->points[i],
                              &laserCloudFullRes->points[i]);
        }

		//����ʱ������ϵ�¶�Ӧ�ĵ�ǰ֡ȫ��������
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
		//Ťת��
        odomAftMapped.twist.twist.angular.x = transformBefMapped[0];
        odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
        odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
        odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
        odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
        odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
        pubOdomAftMapped.publish(odomAftMapped);

        // mappingOdometryFreq.report();
		//�㲥����ϵ��תƽ�Ʋ���
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
