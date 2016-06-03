#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <map>
#include <mutex>
#include <thread>
#include "Logger.hpp"


using PointType = pcl::PointXYZRGBA;
using PointTypeFull = pcl::PointXYZRGBNormal;
using Cloud = pcl::PointCloud <PointType>;
using CloudWithNormals = pcl::PointCloud <PointTypeFull>;

namespace fs = boost::filesystem;

constexpr size_t operator "" _sz (unsigned long long size) {
  return size_t{size};
}

constexpr double operator "" _deg (long double deg) {
  return deg * M_PI / 180.0;
}

constexpr double operator "" _deg (unsigned long long deg) {
  return deg * M_PI / 180.0;
}

constexpr double operator "" _cm (long double cm) {
  return cm / 100.0;
}

constexpr double operator "" _cm (unsigned long long cm) {
  return cm / 100.0;
}

constexpr double operator "" _mm (long double mm) {
  return mm / 1000.0;
}

constexpr double operator "" _mm (unsigned long long mm) {
  return mm / 1000.0;
}

constexpr auto MIN_VALID_ARGS = 3;
constexpr auto MAX_VALID_ARGS = 7;
constexpr auto DEFAULT_Z_DISTANCE = 15_mm;
constexpr auto PLANE_THOLD = 2_cm;
constexpr auto CLUSTER_TOLERANCE = 0.005;
constexpr auto MIN_CLUSTER_SIZE = 100U;
constexpr auto MAX_CLUSTER_SIZE = std::numeric_limits <unsigned int>::max ();
constexpr auto MAX_ITERATIONS_DEFAULT = 1000;
constexpr auto SAC_SIGMA_DEFAULT = 2.0;
constexpr auto REFINE_ITERATIONS_DEFAULT = 50;
constexpr auto MIN_HEIGHT_FROM_HULL_DEFAULT = 0.5_cm;
constexpr auto MAX_HEIGHT_FROM_HULL_DEFAULT = 70_cm;
constexpr auto DEFAULT_HULL_CLOUD_FILENAME = "hull.pcd";

std::mutex deque_mutex;

auto printHelp (int argc, char ** argv) -> void {
  using pcl::console::print_error;
  using pcl::console::print_info;

  print_error ("Syntax is: %s [-z] [-d 0.015] (<path-to-pcd-files> <path-to-store-output>) | "
                   "(<pcd-file> <output-pcd-file> <plane-coords-yml>)| -h | --help\n", argv[0]);
  print_info ("%s -h | --help : shows this help\n", argv[0]);
  print_info ("-d xx : Distance xx in metres from surface below which all points "
                  "will be discarded.\n");
  print_info ("-z : Get all points below the planar surface.\n");
}

auto getPcdFilesInPath (fs::path const & pcd_dir)
-> std::deque <fs::path> {
  auto result_set = std::deque <fs::path>{};
  for (auto const & entry : boost::make_iterator_range (fs::directory_iterator{pcd_dir})) {
    if (fs::is_regular_file (entry.status ())) {
      if (entry.path ().extension () == ".pcd") {
        result_set.emplace_back (entry);
      }
    }
  }
  return result_set;
}

auto expandTilde (std::string path_string) -> fs::path {
  if (path_string.at (0) == '~')
    path_string.replace (0, 1, getenv ("HOME"));
  return fs::path{path_string};
}

auto extractIndices (Cloud::Ptr cloud,
                     pcl::PointIndicesPtr indices,
                     bool keep_organised = false,
                     bool negative = false)
-> Cloud::Ptr {
  auto extract = pcl::ExtractIndices <pcl::PointXYZRGBA>{};
  extract.setInputCloud (cloud);
  extract.setIndices (indices);
  if (cloud->isOrganized ())
    extract.setKeepOrganized (keep_organised);
  extract.setNegative (negative);
  auto extracted_cloud = boost::make_shared <Cloud> ();
  extract.filter (*extracted_cloud);
  return extracted_cloud;
}

// Euclidean Clustering
auto euclideanClustering (Cloud::Ptr cloud,
                          double cluster_tolerance = CLUSTER_TOLERANCE,
                          unsigned int min_size = MIN_CLUSTER_SIZE,
                          unsigned int max_size = MAX_CLUSTER_SIZE)
-> pcl::PointIndicesPtr {
  // kd-tree object for searches.
  auto kd_tree = boost::make_shared <pcl::search::KdTree <PointType>> ();
  kd_tree->setInputCloud (cloud);

  // Euclidean clustering object.
  auto clustering = pcl::EuclideanClusterExtraction <PointType>{};
  // Set cluster tolerance to 1cm (small values may cause objects to be divided
  // in several clusters, whereas big values may join objects in a same cluster).
  clustering.setClusterTolerance (cluster_tolerance);
  // Set the minimum and maximum number of points that a cluster can have.
  clustering.setMinClusterSize (min_size);
  clustering.setMaxClusterSize (max_size);
  clustering.setSearchMethod (kd_tree);
  clustering.setInputCloud (cloud);
  auto clusters = std::vector <pcl::PointIndices>{};
  clustering.extract (clusters);

  if (clusters.size () == 0)
    return nullptr;

  // Find largest cluster and return indices
  auto largest_cluster_index = 0_sz;
  auto current_cluster_index = 0_sz;
  auto max_points = 0_sz;
  for (const auto & cluster: clusters) {
    if (cluster.indices.size () > max_points) {
      max_points = cluster.indices.size ();
      largest_cluster_index = current_cluster_index;
    }
    current_cluster_index++;
  }

  auto indicesPtr = boost::make_shared <pcl::PointIndices> (clusters.at (largest_cluster_index));
  return indicesPtr;
}

auto euclideanClusteringNormals (Cloud::Ptr cloud, double cluster_tolerance)
-> Cloud::Ptr {
  // kd-tree object for searches.
  auto kd_tree = boost::make_shared <pcl::search::KdTree <PointType>> ();
  kd_tree->setInputCloud (cloud);

  auto normals = boost::make_shared <pcl::PointCloud <pcl::PointNormal>> ();
  auto normal_estimation = pcl::NormalEstimation <PointType, pcl::PointNormal>{};
  normal_estimation.setInputCloud (cloud);
  normal_estimation.setRadiusSearch (0.02);
  normal_estimation.setSearchMethod (kd_tree);
  normal_estimation.compute (*normals);

  // kd-tree object for searches.
  auto kd_tree_clusters = boost::make_shared <pcl::search::KdTree <pcl::PointNormal>> ();
  kd_tree_clusters->setInputCloud (normals);

  // Euclidean clustering object.
  auto clustering = pcl::EuclideanClusterExtraction <pcl::PointNormal>{};
  // Set cluster tolerance to 1cm (small values may cause objects to be divided
  // in several clusters, whereas big values may join objects in a same cluster).
  clustering.setClusterTolerance (cluster_tolerance);
  // Set the minimum and maximum number of points that a cluster can have.
  //  clustering.setMinClusterSize(min_size);
  //  clustering.setMaxClusterSize(max_size);
  clustering.setSearchMethod (kd_tree_clusters);
  clustering.setInputCloud (normals);
  auto clusters = std::vector <pcl::PointIndices>{};
  clustering.extract (clusters);

  // Find largest cluster and return indices
  auto largest_cluster_index = 0UL;
  auto current_cluster_index = 0UL;
  auto max_points = 0UL;
  for (const auto & cluster: clusters) {
    if (cluster.indices.size () > max_points) {
      max_points = cluster.indices.size ();
      largest_cluster_index = current_cluster_index;
    }
    current_cluster_index++;
  }

  auto indices_ptr = boost::make_shared <pcl::PointIndices> (clusters.at (largest_cluster_index));
}

auto isSimilarIntensity (const PointType & point_a,
                         const PointType & point_b,
                         float squared_distance)
-> bool {
  auto dist = (point_a.getRGBVector3i () - point_b.getRGBVector3i ()).cast <float> ().norm ();
  return dist < 5.0F;
}

auto isSimilarCurvature (const PointTypeFull & point_a,
                         const PointTypeFull & point_b,
                         float squared_distance)
-> bool {
  auto point_a_normal = Eigen::Map <const Eigen::Vector3f>{point_a.normal};
  auto point_b_normal = Eigen::Map <const Eigen::Vector3f>{point_b.normal};

  return (fabs (point_a_normal.dot (point_b_normal)) < 0.2);
}

auto euclideanClusteringConditional (Cloud::Ptr cloud, double cluster_tolerance)
-> Cloud::Ptr {
  auto kd_tree = boost::make_shared <pcl::search::KdTree <PointType>> ();
  auto cloud_with_normals = boost::make_shared <CloudWithNormals> ();
  pcl::copyPointCloud (*cloud, *cloud_with_normals);
  auto norm_est = pcl::NormalEstimation <PointType, PointTypeFull>{};
  norm_est.setInputCloud (cloud);
  norm_est.setSearchMethod (kd_tree);
  norm_est.setRadiusSearch (1_cm);
  norm_est.compute (*cloud_with_normals);

  auto clustering = pcl::ConditionalEuclideanClustering <PointTypeFull>{true};
  clustering.setInputCloud (cloud_with_normals);
  clustering.setConditionFunction (&isSimilarCurvature);
  clustering.setClusterTolerance (CLUSTER_TOLERANCE);

  auto clusters = std::vector <pcl::PointIndices>{};
  clustering.segment (clusters);

  if (clusters.size () == 0)
    return nullptr;

  // Find largest cluster and return indices
  auto largest_cluster_index = 0UL;
  auto current_cluster_index = 0UL;
  auto max_points = 0UL;
  for (const auto & cluster: clusters) {
    if (cluster.indices.size () > max_points) {
      max_points = cluster.indices.size ();
      largest_cluster_index = current_cluster_index;
    }
    current_cluster_index++;
  }

  auto indices_ptr = boost::make_shared <pcl::PointIndices> (clusters.at (largest_cluster_index));
  return extractIndices (cloud, indices_ptr);
}

auto getLargestRegion (Cloud::Ptr cloud,
                       double smoothness_degrees = 20.0,
                       double curvature_thold = 1.0,
                       double normal_radius_search = 0.02,
                       size_t num_neighbours = 30,
                       size_t min_size = 100,
                       size_t max_size = std::numeric_limits <size_t>::max ())
-> Cloud::Ptr {
  // kd-tree object for searches.
  auto kd_tree = boost::make_shared <pcl::search::KdTree <PointType>> ();
  kd_tree->setInputCloud (cloud);

  // Estimate the normals.
  auto normals = boost::make_shared <pcl::PointCloud <pcl::Normal>> ();
  auto normal_estimation = pcl::NormalEstimation <PointType, pcl::Normal>{};
  normal_estimation.setInputCloud (cloud);
  normal_estimation.setRadiusSearch (normal_radius_search);
  normal_estimation.setSearchMethod (kd_tree);
  normal_estimation.compute (*normals);

  // Region growing clustering object.
  pcl::RegionGrowing <PointType, pcl::Normal> clustering;
  //  clustering.setMinClusterSize((int) min_size);
  //  clustering.setMaxClusterSize((int) max_size);
  clustering.setSearchMethod (kd_tree);
  //  clustering.setNumberOfNeighbours((int) num_neighbours);
  clustering.setInputCloud (cloud);
  clustering.setInputNormals (normals);
  // Set the angle in radians that will be the smoothness threshold
  // (the maximum allowable deviation of the normals).
  clustering.setSmoothnessThreshold ((float) (smoothness_degrees / 180.0 * M_PI)); // degrees.
  // Set the curvature threshold. The disparity between curvatures will be
  // tested after the normal deviation check has passed.
  clustering.setCurvatureThreshold ((float) curvature_thold);

  auto clusters = std::vector <pcl::PointIndices> ();
  clustering.extract (clusters);

  // Find largest cluster and return indices
  auto largest_cluster_index = 0UL;
  auto current_cluster_index = 0UL;
  auto max_points = 0UL;
  for (const auto & cluster: clusters) {
    if (cluster.indices.size () > max_points) {
      max_points = cluster.indices.size ();
      largest_cluster_index = current_cluster_index;
    }
    current_cluster_index++;
  }

  auto indices_ptr = boost::make_shared <pcl::PointIndices> (clusters.at (largest_cluster_index));
  return extractIndices (cloud, indices_ptr);
}

auto getPlaneIndices (Cloud::Ptr cloud,
                      double plane_thold = PLANE_THOLD,
                      int max_iterations = MAX_ITERATIONS_DEFAULT,
                      double refine_sigma = SAC_SIGMA_DEFAULT,
                      int refine_iterations = REFINE_ITERATIONS_DEFAULT)
-> pcl::PointIndicesPtr {
  auto model = boost::make_shared <pcl::SampleConsensusModelPlane <PointType>> (cloud);
  auto sac = pcl::RandomSampleConsensus <PointType>{model, plane_thold};
  sac.setMaxIterations (max_iterations);
  auto inliers_indices_ptr = boost::make_shared <pcl::PointIndices> ();
  auto result = sac.computeModel ();
  sac.getInliers (inliers_indices_ptr->indices);

  if (!result || inliers_indices_ptr->indices.empty ()) {
    pcl::console::print_error ("No planar model found, relax thresholds and continue.");
    return nullptr;
  }

  sac.refineModel (refine_sigma, refine_iterations);

  sac.getSampleConsensusModel ();
  sac.getInliers (inliers_indices_ptr->indices);

  return inliers_indices_ptr;
}

auto getTable (Cloud::Ptr cloud,
               double plane_thold = PLANE_THOLD,
               int max_iterations = MAX_ITERATIONS_DEFAULT,
               double refine_sigma = SAC_SIGMA_DEFAULT,
               int refine_iterations = REFINE_ITERATIONS_DEFAULT)
-> Cloud::Ptr {
  auto point_indices = getPlaneIndices (cloud);

  if (point_indices->indices.size () == 0) {
    pcl::console::print_highlight ("No plane to be found in input pcd.\n");
    return nullptr;
  }

  auto extracted_plane = extractIndices (cloud, point_indices);

  auto largest_cluster_indices = euclideanClustering (extracted_plane);

  return extractIndices (extracted_plane, largest_cluster_indices);
}

/**
 * This function removes the table from cloud and returns the table as a cloud
 */
auto removeTable (Cloud::Ptr & cloud,
                  double plane_thold = PLANE_THOLD,
                  int max_iterations = MAX_ITERATIONS_DEFAULT,
                  double refine_sigma = SAC_SIGMA_DEFAULT,
                  int refine_iterations = REFINE_ITERATIONS_DEFAULT)
-> Cloud::Ptr {
  auto point_indices = getPlaneIndices (cloud);

  if (point_indices->indices.size () == 0) {
    pcl::console::print_highlight ("No plane to be found in input cloud.\n");
    return nullptr;
  }

  auto extracted_plane = extractIndices (cloud, point_indices);

  auto largest_cluster_indices = euclideanClustering (extracted_plane);

  cloud = extractIndices (cloud, point_indices, true);

  return extractIndices (extracted_plane, largest_cluster_indices);
}

auto getTableConvexHull (Cloud::Ptr table_cloud)
-> Cloud::Ptr {
  auto hull = pcl::ConvexHull <PointType>{};
  auto convex_hull_cloud = boost::make_shared <Cloud> ();

  hull.setInputCloud (table_cloud);
  hull.reconstruct (*convex_hull_cloud);

  if (hull.getDimension () != 2 || convex_hull_cloud->size () == 0) {

    pcl::console::print_error ("The input cloud does not represent a planar surface for hull.\n");
    return nullptr;

  } else {
    return convex_hull_cloud;
  }
}

auto getPointsAboveConvexHull (Cloud::Ptr cloud,
                               Cloud::Ptr convex_hull_cloud,
                               double min_height = MIN_HEIGHT_FROM_HULL_DEFAULT,
                               double max_height = MAX_HEIGHT_FROM_HULL_DEFAULT)
-> Cloud::Ptr {
  if (convex_hull_cloud) {
    if (convex_hull_cloud->size () != 0) {
      auto prism = pcl::ExtractPolygonalPrismData <PointType>{};
      prism.setInputCloud (cloud);
      prism.setInputPlanarHull (convex_hull_cloud);
      prism.setHeightLimits (min_height, max_height);
      auto indices_ptr = boost::make_shared <pcl::PointIndices> ();
      prism.segment (*indices_ptr);
      return extractIndices (cloud, indices_ptr, true);
    }
    else {
      pcl::console::print_error ("The input cloud does not represent a planar surface for hull.\n");
      return nullptr;
    }
  }
}

auto removeOutliers (Cloud::Ptr cloud, int mean_k = 50, float sigma_mult = 1.0)
-> Cloud::Ptr {
  auto outlier_remover = pcl::StatisticalOutlierRemoval <PointType>{};
  outlier_remover.setInputCloud (cloud);
  outlier_remover.setMeanK (mean_k);
  outlier_remover.setStddevMulThresh (sigma_mult);

  auto inlier_cloud = boost::make_shared <Cloud> ();
  outlier_remover.filter (*inlier_cloud);

  return inlier_cloud;
}

// Thread-safe deque access
auto getFront (std::deque <fs::path> & paths_set_deque)
-> fs::path {
  auto next_cloud_path = fs::path{};
  std::lock_guard <std::mutex> lock {deque_mutex};
  if (!paths_set_deque.empty ()) {
    next_cloud_path = paths_set_deque.front ();
    paths_set_deque.pop_front ();
    if (paths_set_deque.size () % 10 == 0) {
      Logger::log (Logger::INFO, "%u items remaining in deque.\n", paths_set_deque.size ());
    }
  }
  return next_cloud_path;
}

// TODO: Make thread-safe
auto processHullSegmentation (fs::path input_cloud_path,
                              fs::path output_cloud_path,
                              Cloud::Ptr table_hull_cloud,
                              double min_height_from_hull = MIN_HEIGHT_FROM_HULL_DEFAULT)
-> bool {
  // Load input pcd
  auto input_cloud = boost::make_shared <Cloud> ();
  if (pcl::io::loadPCDFile <pcl::PointXYZRGBA> (input_cloud_path.c_str (), *input_cloud) == -1) {
    pcl::console::print_error ("Failed to load: %s\n", input_cloud_path);
    return false;
  }

  auto
      output_cloud = getPointsAboveConvexHull (input_cloud, table_hull_cloud, min_height_from_hull);

  if (pcl::io::savePCDFileBinaryCompressed <pcl::PointXYZRGBA> (output_cloud_path.string (),
                                                                *output_cloud) == -1) {
    Logger::log (Logger::ERROR, "Failed to save: %s\n", output_cloud_path);
    return false;
  }

  //  Logger::log(Logger::INFO, "Output file %s written.\n", output_cloud_path.c_str());

  return true;
}

auto segmentationRunner (std::deque <fs::path> & paths_deque,
                         fs::path output_dir,
                         Cloud::Ptr table_hull_cloud,
                         double min_height_from_hull = DEFAULT_Z_DISTANCE)
-> bool {
  auto input_pcd_file = getFront (paths_deque);

  while (!paths_deque.empty ()) {
    auto output_pcd_file = output_dir / input_pcd_file.filename ();
    processHullSegmentation (input_pcd_file,
                             output_pcd_file,
                             table_hull_cloud,
                             min_height_from_hull);
    input_pcd_file = getFront (paths_deque);
  }
  Logger::log (Logger::INFO, "Thread exiting.\n");
}

auto threadedHullSegmentation (std::deque <fs::path> & paths_deque,
                               fs::path output_dir,
                               Cloud::Ptr table_hull_cloud,
                               double min_height_from_hull = DEFAULT_Z_DISTANCE)
-> bool {

  auto runners = std::vector <std::shared_ptr <std::thread>>{};
  auto num_threads_to_use = std::thread::hardware_concurrency ();

  for (auto i = 0u; i < num_threads_to_use; ++i) {
    auto runner = std::make_shared <std::thread> (segmentationRunner,
                                                  std::ref (paths_deque),
                                                  output_dir,
                                                  table_hull_cloud,
                                                  min_height_from_hull);
    runners.push_back (runner);
  }

  for (auto i = 0u; i < runners.size (); ++i)
    if (runners.at (i)->joinable ())
      runners.at (i)->join ();

  return true;
}

auto main (int argc, char ** argv) -> int {
  pcl::console::print_highlight (
      "Tool to remove points from a point cloud(s) based on whether they "
          "lie above or below a large planar surface (a table).\n");

  auto help_flag_1 = pcl::console::find_switch (argc, argv, "-h");
  auto help_flag_2 = pcl::console::find_switch (argc, argv, "--help");

  if (help_flag_1 || help_flag_2) {
    printHelp (argc, argv);
    return -1;
  }

  if (argc > MAX_VALID_ARGS || argc < MIN_VALID_ARGS) {
    pcl::console::print_error ("Invalid number of arguments.\n");
    printHelp (argc, argv);
    return -1;
  }

  auto ss = std::stringstream{};
  auto below = pcl::console::find_switch (argc, argv, "-z");
  auto consumed_args = 1;
  if (below) {
    ss << "Extracting all points below planar surface.." << std::endl;
    consumed_args++;
  }

  auto distance = DEFAULT_Z_DISTANCE;
  if (pcl::console::parse_argument (argc, argv, "-d", distance))
    consumed_args += 2;
  ss << "Using distance of " << distance << " meters." << std::endl;

  // Check if we are working with pcd files
  auto pcd_arg_indices = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (pcd_arg_indices.size () == 3) {
    auto input_pcd_file = expandTilde (std::string{argv[pcd_arg_indices.at (0)]});
    auto output_pcd_file = expandTilde (std::string{argv[pcd_arg_indices.at (1)]});
    auto output_table_pcd_file = expandTilde (std::string{argv[pcd_arg_indices.at (2)]});
    consumed_args += 3;
    if (consumed_args != argc) {
      printHelp (argc, argv);
      return -1;
    }

    // Load input pcd
    auto input_cloud = boost::make_shared <Cloud> ();
    if (pcl::io::loadPCDFile <pcl::PointXYZRGBA> (input_pcd_file.c_str (), *input_cloud) == -1) {
      pcl::console::print_error ("Failed to load: %s\n", input_pcd_file);
      printHelp (argc, argv);
      return -1;
    }

    // Print any settings
    pcl::console::print_info (ss.str ().c_str ());

    // Get table
    auto table_cloud = removeTable (input_cloud);

    table_cloud = removeOutliers (table_cloud);
    //    table_cloud = removeOutliers(table_cloud);
    //    table_cloud = euclideanClusteringNormals(table_cloud, 0.1);
    //    table_cloud = getLargestRegion(table_cloud);
    //    table_cloud = euclideanClusteringConditional(table_cloud, CLUSTER_TOLERANCE);

    // Get table hull cloud and get points above table cloud
    auto table_hull = getTableConvexHull (table_cloud);

    if (!table_hull) {
      pcl::console::print_error ("No planar surface in the given cloud");
      return -1;
    }

    auto output_cloud = getPointsAboveConvexHull (input_cloud, table_hull, distance);

    // Visualize both the original and the result.
    auto viewer = pcl::visualization::PCLVisualizer{"Cloud Viewer"};

    // Change colors
    auto table_color_handler =
        pcl::visualization::PointCloudColorHandlerCustom <PointType>{table_cloud, 0, 0, 255};
    auto output_color_handler =
        pcl::visualization::PointCloudColorHandlerCustom <PointType>{output_cloud, 255, 200, 200};
    //    viewer.addPointCloud(table_cloud, "table");
    viewer.addPointCloud (output_cloud, output_color_handler, "output");
    //    viewer.addPointCloud(table_cloud, table_color_handler, "table");
    viewer.addPolygon <PointType> (table_hull, 1.0, 0.0, 0.0);

    viewer.initCameraParameters ();
    pcl::visualization::Camera camera;
    viewer.getCameraParameters (camera);
    camera.view[1] *= -1;
    viewer.setCameraParameters (camera);

    while (!viewer.wasStopped ()) {
      viewer.spinOnce ();
    }
  }
  else {
    // We are working with folders
    auto input_dir = expandTilde (std::string{argv[argc - 2]});
    if (!fs::exists (input_dir) || !fs::is_directory (input_dir)) {
      pcl::console::print_error ("A valid input directory was not specified.\n");
      printHelp (argc, argv);
      return -1;
    }

    auto output_dir = expandTilde (std::string{argv[argc - 1]});
    if (!fs::exists (output_dir) || !fs::is_directory (output_dir)) {
      try {
        fs::create_directory (output_dir);
      } catch (fs::filesystem_error e) {
        pcl::console::print_error ("Unable to create output directory. "
                                       "Please ensure that the correct permissions are "
                                       "set for the target folder and that the path is valid.\n");
        printHelp (argc, argv);
        return -1;
      }
    }

    // We have the input and output dir now
    auto input_files = getPcdFilesInPath (input_dir);

    // Load an input pcd
    auto input_cloud = boost::make_shared <Cloud> ();
    auto mid_point = input_files.size () / 2;
    auto sample_cloud_file = input_files.at (mid_point);
    if (pcl::io::loadPCDFile <pcl::PointXYZRGBA> (sample_cloud_file.c_str (), *input_cloud) == -1) {
      pcl::console::print_error ("Failed to load: %s\n", sample_cloud_file.c_str ());
      printHelp (argc, argv);
      return -1;
    }

    // Print any settings
    pcl::console::print_info (ss.str ().c_str ());

    // Get table
    auto table_cloud = removeTable (input_cloud);

    // Remove Outliers
    table_cloud = removeOutliers (table_cloud);

    // Get table hull cloud and get points above table cloud
    auto table_hull = getTableConvexHull (table_cloud);

    if (!table_hull) {
      pcl::console::print_error ("No planar surface in the given cloud");
      return -1;
    }

    auto table_hull_file = output_dir / DEFAULT_HULL_CLOUD_FILENAME;
    if (pcl::io::savePCDFile (table_hull_file.string (), *table_hull) == -1) {
      pcl::console::print_error ("Failed to save: %s\n", table_hull_file);
      return -1;
    }

    threadedHullSegmentation (input_files, output_dir, table_hull, distance);
  }

  return 0;
}
