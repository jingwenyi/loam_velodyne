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
//����ֲ�����ϵ�µ����еĵ���Ե�һ����ʼ������ڼӼ����˶�������λ�ƻ���
void ShiftToStartIMU(float pointTime) {
//��������ڵ�һ�������ڼӼ��ٲ����Ļ���λ��,��λ������ȫ������ϵ��
//(ȫ������ϵ�»���λ���� delta_Tg)
  imuShiftFromStartXCur =
      imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
  imuShiftFromStartYCur =
      imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
  imuShiftFromStartZCur =
      imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;

//�����ǽ�����λ�ƴ�ȫ������ϵת������������ϵ
//ǰ�����Ǵӻ�������ϵת��ȫ������ϵR = Ry(yaw)*Rx(pitch)*Rz(roll)
//������Ҫ����������ϵ����������ϵ������Ҫ��R ��ת�þ�����Ե�ǰdelta_Tg

//��y ����ת(-imuYawStart), ��Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuShiftFromStartXCur -
             sin(imuYawStart) * imuShiftFromStartZCur;
  float y1 = imuShiftFromStartYCur;
  float z1 = sin(imuYawStart) * imuShiftFromStartXCur +
             cos(imuYawStart) * imuShiftFromStartZCur;
//��x ����ת(-imuPitchStart), ��Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;
//��z ����ת(-imuRollStart), ��Rz(roll).inverse
  imuShiftFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuShiftFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuShiftFromStartZCur = z2;
}
// imu velocity from start vector (imuVeloFromStart*Cur) converted into start
// imu coordinates?
//����ֲ�����ϵ�µ����еĵ�����ڵ�һ����ʼ�����ڼӼ��ٲ������ٶȻ���
void VeloToStartIMU() {
//��������ڵ�һ�������ڼӼ��ٲ����Ļ����ٶȣ����ٶ�Ϊȫ������ϵ
  imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
  imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
  imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;
//�ѻ����ٶ�ת������������ϵ

//��y ����ת(-imuYawStart), ��Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuVeloFromStartXCur -
             sin(imuYawStart) * imuVeloFromStartZCur;
  float y1 = imuVeloFromStartYCur;
  float z1 = sin(imuYawStart) * imuVeloFromStartXCur +
             cos(imuYawStart) * imuVeloFromStartZCur;
//��x ����ת(-imuPitchStart), ��Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;
//��Z ����ת(-imuRollStart), ��Rz(roll).inverse
  imuVeloFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuVeloFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuVeloFromStartZCur = z2;
}
// points converted into start imu coordinates?
//ȥ�����ƼӼ��ٲ�����λ�ƻ���
//��Ϊλ�ƵĻ��������ڵ�һ�����Ƴ��Ļ�������ϵ�²�����
//������Ҫ�ѵ�ǰ���Ƹ��ݵ�ǰ��̬��ת����������ϵ�£�
//��ͨ����ʼ���Ƶ���̬�Ѱ���ת����ʼ���ƶ�Ӧ�Ļ�������ϵ��
//�ټ��ϻ�����
void TransformToStartIMU(PointType *p) {
//ͨ���õ��Ƶĵ�ǰ��̬�ѵ�����ת��ȫ������

//��Z ����ת(imuRollCur)
  float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
  float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
  float z1 = p->z;

//��x ����ת(imupitchCurr)
  float x2 = x1;
  float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
  float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;
//��y ����ת(imuYawCur)
  float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
  float y3 = y2;
  float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

//ͨ����ʼ���Ƶ���̬�Ѹõ���ȫ��������ת����ʼ���ƶ�Ӧ�Ļ�������ϵ��
//�ټ��ϻ���λ�ƣ���ʵ���˿�ʼ���ƺ�������еĵ��ƶ���λ����ʼ���ƵĻ�������ϵ��

//��y �� ��ת(-imuYawStart)
  float x4 = cos(imuYawStart) * x3 - sin(imuYawStart) * z3;
  float y4 = y3;
  float z4 = sin(imuYawStart) * x3 + cos(imuYawStart) * z3;
//��x �� ��ת(-imuPitchStart)
  float x5 = x4;
  float y5 = cos(imuPitchStart) * y4 + sin(imuPitchStart) * z4;
  float z5 = -sin(imuPitchStart) * y4 + cos(imuPitchStart) * z4;

//��z ����ת(-imuRollStart), Ȼ�����ƽ��
  p->x =
      cos(imuRollStart) * x5 + sin(imuRollStart) * y5 + imuShiftFromStartXCur;
  p->y =
      -sin(imuRollStart) * x5 + cos(imuRollStart) * y5 + imuShiftFromStartYCur;
  p->z = z5 + imuShiftFromStartZCur;
}
// compute last shift to imuShift*[imuPointerLast] and velo to
// imuVelo*[imuPointerLast] using previous shift/velo/acc
//ͨ�������zxy ������ļ��ٶȣ������ٶȺ�λ��
void AccumulateIMUShift() {
//��ȡ��������ϵ�µ���̬
  float roll = imuRoll[imuPointerLast];
  float pitch = imuPitch[imuPointerLast];
  float yaw = imuYaw[imuPointerLast];

//����zxy ����ϵ�µ���ʵ���ٶ�
  float accX = imuAccX[imuPointerLast];
  float accY = imuAccY[imuPointerLast];
  float accZ = imuAccZ[imuPointerLast];


//����ǰʱ�̵ļ��ٶ�ֵ�ƽ�������ZXY�̶���(ԭXYZ) �ֱ���ת(roll, pitch, yaw)�Ƕ�
//ת���õ�����������ϵ�µļ��ٶ�ֵ

//��Z ����ת(roll)
  float x1 = cos(roll) * accX - sin(roll) * accY;
  float y1 = sin(roll) * accX + cos(roll) * accY;
  float z1 = accZ;

//��X ����ת(pitch)
  float x2 = x1;
  float y2 = cos(pitch) * y1 - sin(pitch) * z1;
  float z2 = sin(pitch) * y1 + cos(pitch) * z1;

//��Y ����ת(yaw)
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


//���յ������ݣ�velodyne �״�����ϵ��װΪx ����ǰ��y ������z �����ϵ���������ϵ
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg) {

//һ�������ǰ����˼����״���תһ�ܲɼ�������������
  if (!systemInited) {//����ǰ20 ����������
    systemInitCount++;
    if (systemInitCount >= systemDelay) {
      systemInited = true;
    }
    return;
  }

//��¼ÿ��scan �����ʵĵ�Ŀ�ʼ�ͽ�������
  std::vector<int> scanStartInd(
      N_SCANS, 0); // scanStartInd[scanId] is the first point id of scanId
  std::vector<int> scanEndInd(
      N_SCANS, 0); // scanEndInd[scanId] is the last point id of scanId

//��ǰ����ʱ��
  double timeScanCur =
      laserCloudMsg->header.stamp.toSec(); // time point of current scan
  pcl::PointCloud<pcl::PointXYZ>
      laserCloudIn; // input cloud, NaN points removed
//��Ϣת����pcl ���ݴ��
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
  std::vector<int> indices;
  //�Ƴ��յ�
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

//���Ƶ������
  int cloudSize = laserCloudIn.points.size(); // number of cloud points

  //lidar scan ��ʼ�����ת�ǣ�atan2��Χ[-pi, +pi], ������ת����ȡ��������Ϊ
  //velodyne ��˳ʱ����ת��
  float startOri =
      -atan2(laserCloudIn.points[0].y,
             laserCloudIn.points[0]
                 .x); // ori of first point in cloud on origin x-y plane

//lidar scan ���������ת�ǣ���2*pi ʹ������ת����Ϊ2 * pi
  float endOri =
      -atan2(laserCloudIn.points[cloudSize - 1]
                 .y, // ori of last point in clound on origin x-y plane
             laserCloudIn.points[cloudSize - 1].x) +
      2 * M_PI;

//������λ���뿪ʼ��λ�ǲ�ֵ������(PI, 3*PI) ��Χ������lidar ����һ��Բ��ɨ��
//����������������Χ��: pi < endOri - startOri < 3*pi, �쳣������
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }

//lidar ɨ���Ƿ���ת����
//�����Ǽ��һ��һ�����ݰ��е�һ�����Ƶ�λ�ã��Ƿ���ת����
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
  //����imu ���ݽ�ԭʼɨ�����׼�������״�����ϵ��
  //�Ͱ�ÿ���㶼ͳһ����ʼ���ƵĻ�������ϵ��
  //Ŀ����ȥ�������״�������˶������Ļ���
  //�����״�ɨ��һȦ��Ҫʱ�䣬������ʱ�򼤹��״���Ե���������˶�
  //�ͻᵼ�º���ĵ��Ƹ���һ�����ƻ���������һ���Ļ���
  for (int i = 0; i < cloudSize; i++) {
  	//�����ύ����velodyne lidar������ϵҲת����z ����ǰ��x ���������������ϵ
    point.x = laserCloudIn.points[i].y;
    point.y = laserCloudIn.points[i].z;
    point.z = laserCloudIn.points[i].x;

    // minP.x = std::min(minP.x, point.x);
    // minP.y = std::min(minP.y, point.y);
    // minP.z = std::min(minP.z, point.z);
    // maxP.x = std::max(maxP.x, point.x);
    // maxP.y = std::max(maxP.y, point.y);
    // maxP.z = std::max(maxP.z, point.z);

	//����������(����lidar �ĵ���ֱ�Ǽ��㹫ʽ)�������������м����ߺţ�velodyne
	//ÿ����scan ֮����2 ��
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
	//������������(�Ӽ�0.5 �ض�Ч��������������)
    int roundedAngle = int(angle + (angle < 0.0 ? -0.5 : +0.5));

	//�Ѹõ��Ʒŵ�16 ����ȥ
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

	//���˵㣬ֻ��ѡ[-15�ȣ�+15��] ��Χ�ڵĵ㣬scanID ����[0, 15]
    if (scanID > (N_SCANS - 1) || scanID < 0) { // drop the points with invalid scanIDs
      count--;
      continue;
    }

    const int debug_errorPointIDStart = 121513;
    // if (i >= debug_errorPointIDStart) {
    //   ROS_INFO("point %i's scanID = %i", i, scanID);
    // }    

	//�õ����ת��
    float ori = -atan2(point.x, point.z);

	//����ɨ�����Ƿ���ת����ѡ������ʼλ�û�����ֹλ�ý��в�ֵ���㣬�Ӷ����в���
    if (!halfPassed) {
	//ȷ�� -pi/2 < ori - startOri < 3*pi/2
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

	//ȷ�� -3*pi/2 < ori - endOri < pi/2
      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      }
    }

	//-0.5 < relTime < 1.5, ����ת�ĽǶ�������������ת�Ƕȵı��ʣ��������е�����ʱ��
    float relTime = (ori - startOri) / (endOri - startOri);
	//��ǿ��= �ߺ�+ �����ʱ��(��һ������ +  һ��С���������������ߺţ�С�������Ǹõ�
	//�����ʱ��)�� ����ɨ��:���ݵ�ǰɨ��ĽǶȺ�ɨ������ڼ������ɨ����ʼλ�õ�ʱ��

	//���������������������ǣ�С����������ת�Ǻ����ڵı���
	point.intensity = scanID + scanPeriod * relTime;

    // if (i >= debug_errorPointIDStart) {
    //   ROS_INFO("halfPassed = %i, ori = %f, point intensity = %f", halfPassed,
    //            ori, point.intensity);
    // }

	//��ʱ��= ����ʱ��+ ����ʱ��
	//����յ�imu ���ݣ�ʹ��IMU �������ƻ���
    if (imuPointerLast >= 0) {
	//����������ʱ��
      float pointTime = relTime * scanPeriod;
	//Ѱ���Ƿ��е��Ƶ�ʱ���С��IMU ��ʱ�����IMUλ��:imuPointerFront
      while (imuPointerFront != imuPointerLast) {
        if (timeScanCur + pointTime < imuTime[imuPointerFront]) {
          break;
        }
		//�ڻ���buffer �л�ȡ��һ�����index
        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
      }

      if (timeScanCur + pointTime >
          imuTime[imuPointerFront]) { /// use the newest imu data if no newer
                                      /// imu
        //û�ҵ�����ʱimuPointerFront==imtPointerLast, ֻ���Ե�ǰ�յ������µ�imu ���ٶȣ�λ�ƣ�ŷ��
        //����Ϊ��ǰ����ٶȣ�λ�ƣ�ŷ����ʹ��
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
       //�ҵ��˵���ʱ���С��imu ʱ�����IMUλ�ã���õ�ش���imuPointerBack
       //��imuPointerFront ֮�䣬�ݴ����Բ�ֵ��������Ƶ��ٶȣ�λ�ƺ�ŷ����

		//��ȡ��imuPointerFront ���һ��imu
        int imuPointerBack =
            (imuPointerFront + imuQueLength - 1) % imuQueLength;
		//��ʱ��������Ȩ�ط�����ʣ� Ҳ��ʱ���Բ�ֵ
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
	  	//����ǵ�һ�㣬��ס������ʼλ�õ��ٶȣ�λ�ƣ�ŷ����
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
      //����֮��ÿ��������ڵ�һ��������ڼӼ��ٷ������˶�������λ���ٶȻ��䣬
      //���Ե����е�ÿ����λ����Ϣ���²�������
        ShiftToStartIMU(pointTime);
        VeloToStartIMU();
        TransformToStartIMU(&point);
      }
    }

	//��ÿ�����������ĵ�����Ӧ�ߺŵ�������
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

//�����Ч��Χ�ڵĵ������
  cloudSize = count;

  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  //�����еĵ㰴���ߺŴ�С�������һ������
  for (int i = 0; i < N_SCANS; i++) {
    *laserCloud += laserCloudScans[i];
  }
  int scanCount = -1;
  //ʹ��ÿ�����ǰ�������������ʣ����ǰ�����ͺ����������
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
	//���ʼ���
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
	//��¼���ʵ������
    cloudSortInd[i] = i;
	//��ʼʱ����ȫδɸѡ��
    cloudNeighborPicked[i] = 0;
	//��ʼ��Ϊless flat ��
    cloudLabel[i] = 0;

	//ÿ��scan(�ߺ�), ֻ�е�һ�����ϵĵ���������Ϊÿ��scan �ĵ㶼����һ����
	//����ȡ�����������־��ǵ�ǿ�ȵ��ߺ�
    if (int(laserCloud->points[i].intensity) != scanCount) {
	//����ÿ��scan ֻ�����һ����,  ����ȡ����
      scanCount = int(laserCloud->points[i].intensity);

		//����ֻȡͬһ��scan (�ߺ�) ��������ģ���scan ��������ʷǷ���
		//�ų���Ҳ���ų�ÿ��scan ��ǰ�������
      if (scanCount > 0 && scanCount < N_SCANS) {
        scanStartInd[scanCount] = i + 5;
        scanEndInd[scanCount - 1] = i - 5;
      }
    }
  }
  //��һ��scan ���ʵ���Ч����ӵ�5 ����ʼ���һ�������߽������� size -5
  scanStartInd[0] = 5;
  scanEndInd.back() = cloudSize - 5;

  //ROS_INFO("cloudCurvature scanStartInd scanEndInd computed");



//��ѡ�㣬�ų����ױ�б�浲ס�ĵ��Լ���Ⱥ�㣬��Щ�����ױ�б�浲ס
//����Ⱥ����ܳ�����żȻ�ԣ���Щ��������ܵ���ǰ������ɨ�費�ܱ�ͬʱ����

  for (int i = 5; i < cloudSize - 6; i++) {//���һ����Ĳ�ֵ�����Լ�6
    float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
    float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
    float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
	//������Ч���ʵ����һ����֮��ľ���ƽ����
    float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;

    if (diff > 0.1) {//ǰ��������֮�����Ҫ����0.1

	//������
      float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x +
                          laserCloud->points[i].y * laserCloud->points[i].y +
                          laserCloud->points[i].z * laserCloud->points[i].z);
	//��һ��������
      float depth2 =
          sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x +
               laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
               laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);

	//�������������ȱ���������Ƚϴ�ĵ���������������
	//�������˼�ǰѳ��ı�ͶӰ���̵ı���
      if (depth1 > depth2) {
        diffX = laserCloud->points[i + 1].x -
                laserCloud->points[i].x * depth2 / depth1;
        diffY = laserCloud->points[i + 1].y -
                laserCloud->points[i].y * depth2 / depth1;
        diffZ = laserCloud->points[i + 1].z -
                laserCloud->points[i].z * depth2 / depth1;

		//�߳���Ҳ��ʱ����ֵ����С��0.1�� ˵���нǱȽ�С��б��Ƚ϶��ͣ�����ȱ仯�Ƚ�
		//���ң��㴦�ڽ����뼤����ƽ�е�б����
        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 <
            0.1) { // is connected?
         //�ų����ױ�б�浲ס�ĵ�
         //�õ㼰ǰ�������( ���¶���б���� )  ȫ����Ϊɸѡ��
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
          //�ų����ױ�б�浲ס�ĵ�
         //�õ㼰ǰ�������� ( ���¶���б���� )  ȫ����Ϊɸѡ��
          cloudNeighborPicked[i + 1] = 1;
          cloudNeighborPicked[i + 2] = 1;
          cloudNeighborPicked[i + 3] = 1;
          cloudNeighborPicked[i + 4] = 1;
          cloudNeighborPicked[i + 5] = 1;
          cloudNeighborPicked[i + 6] = 1;
        }
      }
    }

	//��ǰ���ǰһ��ľ����
    float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
    float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
    float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
	//������ƽ��
    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

	//��ǰ�����ȵ�ƽ��
    float dis = laserCloud->points[i].x * laserCloud->points[i].x +
                laserCloud->points[i].y * laserCloud->points[i].y +
                laserCloud->points[i].z * laserCloud->points[i].z;

	//��ǰ����ƽ���Ͷ��������ƽ���͵����֮������Щ����Ϊ��Ⱥ�㣬������б����
	//�ĵ㣬ǿ�Ұ�͹��Ϳտ������е�ĳЩ�㣬����Ϊɸѡ��������
    if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) {
      cloudNeighborPicked[i] = 1;
    }
  }

  //ROS_INFO("cloudNeighborPicked initialized");

  pcl::PointCloud<PointType> cornerPointsSharp;     // the outputs
  pcl::PointCloud<PointType> cornerPointsLessSharp; // the outputs
  pcl::PointCloud<PointType> surfPointsFlat;        // the outputs
  pcl::PointCloud<PointType> surfPointsLessFlat;    // the outputs

	//��ÿ�����ϵĵ������Ӧ�����: ���ص��ƽ���������
  for (int i = 0; i < N_SCANS; i++) {
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(
        new pcl::PointCloud<PointType>);
	//��ÿ��scan �����ʵ�ֳ�6 �ȷݴ���ȷ����Χ���е㱻ѡ��
    for (int j = 0; j < 6; j++) {
		//���ȷݵ����: sp = scanStartInd + (scanEndInd - scanStartInd) * j / 6
      int sp = (scanStartInd[i] * (6 - j) + scanEndInd[i] * j) / 6;
		//���ȷݵ��յ�: ep = scanStartInd -1 + (scanEndInd - scanStartInd)*(j+1) / 6 -1
      int ep = (scanStartInd[i] * (5 - j) + scanEndInd[i] * (j + 1)) / 6 - 1;

	//�����ʴ�С����ð������
      for (int k = sp + 1; k <= ep; k++) { // sort by curvature within [sp,
                                           // ep]?, curvature descending order
        for (int l = k; l >= sp + 1; l--) {
			//����������ʴ���ǰ���򽻻�
          if (cloudCurvature[cloudSortInd[l]] <
              cloudCurvature[cloudSortInd[l - 1]]) {
            int temp = cloudSortInd[l - 1];
            cloudSortInd[l - 1] = cloudSortInd[l];
            cloudSortInd[l] = temp;
          }
        }
      }

	//��ѡÿ���ֶε����ʺܴ�ͱȽϴ�ĵ�
      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSortInd[k];//��������ĵ���
        //������ʴ�ĵ㣬���ʵ�ȷ�Ƚϴ󣬲���δ��ɸѡ���˵�
        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > 0.1) {

          largestPickedNum++;
		  //��ѡ��������ǰ2 �������sharp �㼯��
          if (largestPickedNum <= 2) {
            cloudLabel[ind] = 2;// 2 �������ʺܴ�
            cornerPointsSharp.push_back(laserCloud->points[ind]);
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else if (largestPickedNum <= 20) {
          //��ѡ��������ǰ20 �������less sharp �㼯��
            cloudLabel[ind] = 1;// 1 �������ʱȽϼ���
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else {
            break;
          }

		//ɸѡ��־��λ
          cloudNeighborPicked[ind] = 1;

		//�����ʱȽϴ�ĵ��ǰ���5 ����������ȽϽ��ĵ�ɸѡ��ȥ����ֹ������ۼ���
		//ʹ����������ÿ�������Ͼ����ֲ�����
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
	//��ѡÿ���ֶε����ʺ�С�Ƚ�С�ĵ�
      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];// ������С��ĵ���
        //������ʵ�ȷ�Ƚ�С������δ��ɸѡ��
        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0.1) {

          cloudLabel[ind] = -1;//-1 �������ʺ�С�ĵ�
          surfPointsFlat.push_back(laserCloud->points[ind]);

          smallestPickedNum++;
		  //ֻѡ��С���ĸ���ʣ���label = 0, �������ʱȽ�С��
          if (smallestPickedNum >= 4) {
            break;
          }
		//ɸѡ��־����
          cloudNeighborPicked[ind] = 1;

		//ͬ����ֹ������ۼ�
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

	//��ʣ��ĵ�(����֮ǰ���ų��ĵ�) ȫ������ƽ����� less plat �����
      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0) {
          surfPointsLessFlatScan->push_back(laserCloud->points[k]);
        }
      }
    }

	//����less flat ����࣬��ÿ���ֶ�less flat �ĵ��������դ���˲�
    pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
	//���������˲����Ķ���
    pcl::VoxelGrid<PointType> downSizeFilter;
	//������Ҫ�˲��ĵ���
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
	//����դ���С����Ϊ��20*20*20cm
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(surfPointsLessFlatScanDS);

	//less flat �����
    surfPointsLessFlat += surfPointsLessFlatScanDS;
  }

  //ROS_INFO("feature points collected");

//publish �����������˶����������е�
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/camera";
  pubLaserCloud.publish(laserCloudOutMsg);

//publish �����������˶�������ƽ���ͱ��ص�
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
//publish imu ��Ϣ������ѭ��������������cur ���Ǵ������һ����
//�����һ�����ŷ���ǣ�����λ�Ƽ�һ�������������ӵ��ٶ�
  pcl::PointCloud<pcl::PointXYZ> imuTrans(4, 1);
//��ʼ���ŷ����
  imuTrans.points[0].x = imuPitchStart;
  imuTrans.points[0].y = imuYawStart;
  imuTrans.points[0].z = imuRollStart;

//���һ�����ŷ����
  imuTrans.points[1].x = imuPitchCur;
  imuTrans.points[1].y = imuYawCur;
  imuTrans.points[1].z = imuRollCur;
//���һ��������ڵ�һ����Ļ���λ�ƺ��ٶ�
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

	//��������ϵ��������Ӱ��
  float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;
  float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
  float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

  //#define PRINT(name) ROS_INFO(#name" = %f\n", name)
  //  PRINT(accX);
  //  PRINT(accY);
  //  PRINT(accZ);
  //#undef PRINT

//ѭ����λЧ�����γɻ�������
  imuPointerLast = (imuPointerLast + 1) % imuQueLength;

//��¼imu ʱ�������̬(��������ϵ��)
  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = roll;
  imuPitch[imuPointerLast] = pitch;
  imuYaw[imuPointerLast] = yaw;

  //��¼��ȥ����Ӱ���zxy ����ϵ��������ļ��ٶ�
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
