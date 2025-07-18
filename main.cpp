#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <PCL/features/normal_3d.h>
#include "viz/PC_viz.h"
#include "RRHT/PCL_normEst.h"

int main(int argc, char **argv)
{
    std::filesystem::path folderPath = std::filesystem::current_path() / "data" / "PCD";
    std::vector<std::string> Files;

    // PCD 파일 수집
    for (const auto &entry : std::filesystem::directory_iterator(folderPath))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".pcd")
            Files.push_back(entry.path().string());
    }

    if (Files.empty())
    {
        std::cerr << "No PCD files found in: " << folderPath << std::endl;
        return -1;
    }

    for (const auto &file : Files)
    {
        //---------------------
        //* 1. PCD 파일 로드 및 다운샘플링
        //---------------------
        std::cout << "Processing: " << file << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (pcl::io::loadPCDFile(file, *cloud) == -1)
        {
            PCL_ERROR("Failed to load: %s\n", file.c_str());
            continue;
        }

        //---------------------
        //* 2. RRHT 법선 추정 및 PCL 기본 법선 추정
        //---------------------
        pcl::PointCloud<pcl::Normal>::Ptr rrht_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr pcl_normals(new pcl::PointCloud<pcl::Normal>);

        //----------------------
        //* 2-1. RRHT 법선 추정
        //----------------------
        PCL_Normal_Estimator<pcl::PointXYZ, pcl::Normal> normal_estimator(cloud, rrht_normals);
        normal_estimator.number_of_planes() = 2000;
        normal_estimator.rotation_number() = 3;
        normal_estimator.accum_slices() = 30;
        normal_estimator.cluster_angle_rad() = 0.1;
        normal_estimator.small_radius_fact() = 4;
        normal_estimator.normal_selection_mode() = PCL_Normal_Estimator<pcl::PointXYZ, pcl::Normal>::CLUSTER;
        normal_estimator.minimal_neighbor_number_for_range_search() = 40;

        auto start1 = std::chrono::high_resolution_clock::now();
        normal_estimator.estimate(
            PCL_Normal_Estimator<pcl::PointXYZ, pcl::Normal>::CUBES,
            PCL_Normal_Estimator<pcl::PointXYZ, pcl::Normal>::KNN,
            cloud->points.size() * 0.6);

        //------------------------
        //* 2-2. PCL 기본 법선 추정
        //------------------------
        auto start2 = std::chrono::high_resolution_clock::now();
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(tree);
        ne.setKSearch(30);
        ne.setInputCloud(cloud);
        ne.compute(*pcl_normals);

        pcl::PointCloud<pcl::PointXYZ>::Ptr valid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::Normal>::Ptr valid_rrht_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr valid_pcl_normals(new pcl::PointCloud<pcl::Normal>);

        for (size_t i = 0; i < cloud->size(); ++i)
        {
            if (!std::isnan(rrht_normals->points[i].normal_x) &&
                !std::isnan(rrht_normals->points[i].normal_y) &&
                !std::isnan(rrht_normals->points[i].normal_z) &&
                !std::isnan(pcl_normals->points[i].normal_x) &&
                !std::isnan(pcl_normals->points[i].normal_y) &&
                !std::isnan(pcl_normals->points[i].normal_z))
            {
                valid_cloud->push_back(cloud->points[i]);
                valid_rrht_normals->push_back(rrht_normals->points[i]);
                valid_pcl_normals->push_back(pcl_normals->points[i]);
            }
        }

        double total_angle_diff = 0.0;
        for (size_t j = 0; j < valid_rrht_normals->size(); ++j)
        {
            Eigen::Vector3d rrht_n(valid_rrht_normals->points[j].normal_x,
                                   valid_rrht_normals->points[j].normal_y,
                                   valid_rrht_normals->points[j].normal_z);

            Eigen::Vector3d pcl_n(valid_pcl_normals->points[j].normal_x,
                                  valid_pcl_normals->points[j].normal_y,
                                  valid_pcl_normals->points[j].normal_z);

            double cos_theta = rrht_n.dot(pcl_n) / (rrht_n.norm() * pcl_n.norm());
            double angle_diff = acos(std::max(-1.0, std::min(1.0, cos_theta))) * (180.0 / M_PI); // degree 변환
            total_angle_diff += angle_diff;
        }
        double avg_angle_diff = total_angle_diff / valid_rrht_normals->size();
        std::cout << "평균 각도 차이: " << avg_angle_diff << " 도\n";

        //----------------------
        //* 3. 시각화
        //----------------------
        Base3DViz viz;
        Base3DViz::PCOptions options;
        options.name = "PCL";
        options.pointSize = 3.0f;
        options.color = {0, 0, 255};
        options.centerCamera = true;
        options.showCentroid = true;
        viz.normal(cloud, pcl_normals, options);

        options.name = "RRHT";
        options.color = {255, 0, 0};
        viz.normal(cloud, rrht_normals, options);
    }
    return 0;
}
