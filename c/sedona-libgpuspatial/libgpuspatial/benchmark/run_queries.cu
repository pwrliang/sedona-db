#include <memory>
#include "../include/gpuspatial/index/spatial_joiner.cuh"
#include "../test/array_stream.hpp"
#include "arrow/api.h"
#include "arrow/c/bridge.h"
#include "arrow/filesystem/api.h"
#include "arrow/filesystem/s3fs.h"
#include "arrow/record_batch.h"
// #include "baseline/boost_index.hpp"

#include "baseline/geos_index.hpp"
#include "geoarrow_geos/geoarrow_geos.h"
#include "geos/vend/json.hpp"
#include "gpuspatial/utils/stopwatch.h"
#include "nanoarrow/nanoarrow.hpp"
#include "nlohmann/json.hpp"
#include "parquet/arrow/reader.h"

#include "run_queries.cuh"

#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#define MAX_DOWNLOAD_THREADS (10)
#define ARROW_THROW_NOT_OK(status_expr)       \
  do {                                        \
    arrow::Status _s = (status_expr);         \
    if (!_s.ok()) {                           \
      throw std::runtime_error(_s.message()); \
    }                                         \
  } while (0)

namespace gpuspatial {

// Helper function to check if a string ends with a specific suffix
static bool HasSuffix(const std::string& str, const std::string& suffix) {
  if (str.length() >= suffix.length()) {
    return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
  }
  return false;
}

bool StartsWith(const std::string& mainString, const std::string& prefix) {
  // If the prefix is longer than the main string, it can't be a prefix.
  if (prefix.length() > mainString.length()) {
    return false;
  }
  // Extract a substring from mainString starting at index 0
  // with the length of the prefix, and compare it.
  return mainString.substr(0, prefix.length()) == prefix;
}

arrow::Status ReadParquetFromS3(
    arrow::fs::FileSystem* fs, const std::string& url, int64_t batch_size,
    std::vector<std::shared_ptr<arrow::Array>>& record_batches,
    int max_files = std::numeric_limits<int>::max()) {
  // 1. Initialize S3 Filesystem

  // std::cout << "Successfully initialized S3 filesystem." << std::endl;

  // 2. Create a selector to list all files recursively
  arrow::fs::FileSelector selector;
  selector.base_dir = url;
  selector.recursive = true;

  // 3. Get the list of FileInfo objects
  ARROW_ASSIGN_OR_RAISE(auto file_infos, fs->GetFileInfo(selector));
  std::cout << "Found " << file_infos.size() << " total objects in " << url << std::endl;

  ThreadPool pool(MAX_DOWNLOAD_THREADS);
  std::vector<std::future<std::vector<std::shared_ptr<arrow::Array>>>> loading_rb;

  int n_read = 0;
  // 4. Iterate through files, filter for Parquet, and read them
  for (const auto& file_info : file_infos) {
    // Skip directories (which are just prefixes in S3)
    if (file_info.type() != arrow::fs::FileType::File) {
      continue;
    }

    const std::string& path = file_info.path();

    // Optional: Filter for files with a .parquet extension
    if (!HasSuffix(path, ".parquet")) {
      std::cout << "  - Skipping non-parquet file: " << path << std::endl;
      continue;
    }
    std::cout << "--- Processing Parquet file: " << path << " ---" << std::endl;

    loading_rb.emplace_back(pool.enqueue(
        [fs, batch_size](
            const arrow::fs::FileInfo& fi) -> std::vector<std::shared_ptr<arrow::Array>> {
          // Open a stream to the S3 file
          auto input_file = fs->OpenInputFile(fi);

          // Open the Parquet file reader
          auto arrow_reader = parquet::arrow::OpenFile(input_file.ValueOrDie(),
                                                       arrow::default_memory_pool())
                                  .ValueOrDie();

          arrow_reader->set_batch_size(batch_size);

          std::shared_ptr<::arrow::RecordBatchReader> rb_reader;
          ARROW_THROW_NOT_OK(arrow_reader->GetRecordBatchReader(&rb_reader));
          std::vector<std::shared_ptr<arrow::Array>> local_record_batches;

          while (true) {
            std::shared_ptr<arrow::RecordBatch> batch;
            ARROW_THROW_NOT_OK(rb_reader->ReadNext(&batch));
            if (!batch) {
              break;
            }
            local_record_batches.push_back(batch->GetColumnByName("geometry"));
          }
          return local_record_batches;
        },
        file_info));

    if (++n_read >= max_files) {
      break;
    }
  }

  for (auto& fu : loading_rb) {
    auto vec = fu.get();
    record_batches.insert(record_batches.end(), vec.begin(), vec.end());
  }

  return arrow::Status::OK();
}

template <typename FUNC_T>
arrow::Status ReadParquetFromS3(arrow::fs::FileSystem* fs, const std::string& url,
                                FUNC_T fn,
                                int max_files = std::numeric_limits<int>::max()) {
  // 1. Initialize S3 Filesystem

  arrow::fs::S3Options options = arrow::fs::S3Options::Defaults();

  ARROW_ASSIGN_OR_RAISE(auto s3fs, arrow::fs::S3FileSystem::Make(options));
  std::cout << "Successfully initialized S3 filesystem." << std::endl;

  // 2. Create a selector to list all files recursively
  arrow::fs::FileSelector selector;
  selector.base_dir = url;
  selector.recursive = true;

  // 3. Get the list of FileInfo objects
  ARROW_ASSIGN_OR_RAISE(auto file_infos, s3fs->GetFileInfo(selector));
  std::cout << "Found " << file_infos.size() << " total objects in " << url << std::endl;

  int n_read = 0;
  // 4. Iterate through files, filter for Parquet, and read them
  for (const auto& file_info : file_infos) {
    // Skip directories (which are just prefixes in S3)
    if (file_info.type() != arrow::fs::FileType::File) {
      continue;
    }

    const std::string& path = file_info.path();

    // Optional: Filter for files with a .parquet extension
    if (!HasSuffix(path, ".parquet")) {
      std::cout << "  - Skipping non-parquet file: " << path << std::endl;
      continue;
    }

    std::cout << "--- Processing Parquet file: " << path << " ---" << std::endl;

    // Open a stream to the S3 file
    ARROW_ASSIGN_OR_RAISE(auto input_file, s3fs->OpenInputFile(file_info));

    // Open the Parquet file reader
    ARROW_ASSIGN_OR_RAISE(
        std::unique_ptr<parquet::arrow::FileReader> arrow_reader,
        parquet::arrow::OpenFile(input_file, arrow::default_memory_pool()));

    arrow_reader->set_batch_size(200 * 1000);

    std::shared_ptr<::arrow::RecordBatchReader> rb_reader;
    ARROW_RETURN_NOT_OK(arrow_reader->GetRecordBatchReader(&rb_reader));

    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      ARROW_RETURN_NOT_OK(rb_reader->ReadNext(&batch));
      if (!batch) {
        break;
      }
      auto geometries = batch->GetColumnByName("geometry");

      nanoarrow::UniqueArray unique_array;
      nanoarrow::UniqueSchema unique_schema;

      auto status =
          arrow::ExportArray(*geometries, unique_array.get(), unique_schema.get());
      if (status.ok()) {
        fn(unique_array.get(), unique_schema.get());
      } else {
        return status;
      }
    }
    if (++n_read >= max_files) {
      break;
    }
  }

  return arrow::Status::OK();
}

static bool append_vector_pairs_to_file(const std::string& file_path,
                                        const std::vector<uint32_t>& vec1,
                                        const std::vector<uint32_t>& vec2) {
  // Check if the vectors have the same size to form pairs.
  if (vec1.size() != vec2.size()) {
    std::cerr << "Error: Vectors must be of the same size to form pairs." << std::endl;
    return false;
  }

  // Open the file in append mode (`std::ios::app`).
  std::ofstream output_file(file_path, std::ios::app);

  // Check if the file was successfully opened.
  if (!output_file.is_open()) {
    std::cerr << "Error: Unable to open file for appending: " << file_path << std::endl;
    return false;
  }

  // Iterate through the vectors and write the pairs.
  for (size_t i = 0; i < vec1.size(); ++i) {
    output_file << vec1[i] << " " << vec2[i] << "\n";
  }

  // The file is automatically closed when `output_file` goes out of scope.
  return true;
}
static ArrowSchema MakeSchemaWKB() {
  ArrowSchema tmp_schema;

  tmp_schema.release = nullptr;

  int result = GeoArrowGEOSMakeSchema(GEOARROW_GEOS_ENCODING_WKB, 0, &tmp_schema);
  if (result != GEOARROW_GEOS_OK) {
    throw std::runtime_error("GeoArrowGEOSMakeSchema failed");
  }
  return tmp_schema;
}

void RunQueries(BenchmarkConfig& config) {
  ARROW_THROW_NOT_OK(arrow::fs::InitializeS3(arrow::fs::S3GlobalOptions()));
  rmm::mr::cuda_async_memory_resource cuda_mr;
  rmm::mr::binning_memory_resource mr(&cuda_mr);
  rmm::mr::set_current_device_resource(&mr);
  std::unique_ptr<StreamingJoiner> joiner;
  Stopwatch sw;
  nlohmann::json json;

  auto& json_config = json["config"];
  auto& json_download = json["download"];
  auto& json_load = json["push_build"];
  auto& json_build = json["build"];
  auto& json_query = json["query"];

  cudaDeviceProp prop;
  int device;

  if (cudaGetDevice(&device) == cudaSuccess) {
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    json_config["gpu_id"] = device;
    json_config["gpu_name"] = prop.name;
  }

  json_config["build_file"] = config.build_file;
  json_config["stream_file"] = config.stream_file;
  json_config["output_file"] = config.output_file;
  json_config["batch_size"] = config.batch_size;
  switch (config.execution) {
    case Execution::kBoost: {
      json_config["execution"] = "Boost";
      break;
    }
    case Execution::kGEOS:
      json_config["execution"] = "GEOS";
      break;
    case Execution::kRT_Filter: {
      json_config["execution"] = "RTFilter";
      break;
    }
    case Execution::kRT: {
      json_config["execution"] = "RT";
      break;
    }
  }
  json_config["limit"] = config.limit;
  json_config["parallelism"] = config.parallelism;
  std::shared_ptr<StreamingJoiner::Config> joiner_config;

  switch (config.execution) {
    case Execution::kBoost:
      // joiner = std::make_unique<BoostIndex>();
      break;
    case Execution::kRT: {
      auto joiner_config_impl = std::make_shared<SpatialJoiner::SpatialJoinerConfig>();
      joiner_config_impl->ptx_root = config.ptx_root.c_str();
      joiner = std::make_unique<SpatialJoiner>();
      joiner_config = joiner_config_impl;

      break;
    }
    case Execution::kGEOS: {
      joiner = std::make_unique<GEOSIndex>();
      break;
    }
    default:
      throw std::runtime_error("Unsupported execution type");
  }
  double push_build_ms = 0, build_index_ms = 0, query_ms = 0;
  int64_t build_size = 0;

  joiner->Init(joiner_config.get());
  ArrowSchema tmp_schema = MakeSchemaWKB();

  std::shared_ptr<arrow::fs::FileSystem> build_fs, stream_fs;

  if (StartsWith(config.build_file, "s3://")) {
    arrow::fs::S3Options options = arrow::fs::S3Options::Defaults();
    auto res = arrow::fs::S3FileSystem::Make(options);
    build_fs = res.ValueOrDie();
    config.build_file.erase(0, std::string("s3://").length());
  } else {
    build_fs = std::make_shared<arrow::fs::LocalFileSystem>();
  }

  if (StartsWith(config.stream_file, "s3://")) {
    arrow::fs::S3Options options = arrow::fs::S3Options::Defaults();
    auto res = arrow::fs::S3FileSystem::Make(options);
    stream_fs = res.ValueOrDie();
    config.stream_file.erase(0, std::string("s3://").length());
  } else {
    stream_fs = std::make_shared<arrow::fs::LocalFileSystem>();
  }

  {
    sw.start();
    std::vector<std::shared_ptr<arrow::Array>> record_batches;
    ARROW_THROW_NOT_OK(
        ReadParquetFromS3(build_fs.get(), config.build_file, 256 * 1000, record_batches));
    sw.stop();
    json_download["build_file_time"] = sw.ms();

    for (size_t i = 0; i < record_batches.size(); i++) {
      auto& batch = record_batches[i];
      nanoarrow::UniqueArray unique_array;
      nanoarrow::UniqueSchema unique_schema;
      nlohmann::json json_batch;

      ARROW_THROW_NOT_OK(
          arrow::ExportArray(*batch, unique_array.get(), unique_schema.get()));

      sw.start();
      joiner->PushBuild(&tmp_schema, unique_array.get(), 0, unique_array->length);
      sw.stop();
      json_batch["record_batch_idx"] = i;
      json_batch["size"] = unique_array->length;
      json_batch["time"] = sw.ms();
      json_load.push_back(json_batch);
      push_build_ms += sw.ms();
      build_size += unique_array->length;
    }
  }

  sw.start();
  joiner->FinishBuilding();
  sw.stop();
  json_build["time"] = sw.ms();
  json_build["size"] = build_size;
  build_index_ms = sw.ms();

  printf("Number of polygons %ld, Push Build %lf ms, Build Index %f ms\n", build_size,
         push_build_ms, build_index_ms);

  sw.start();
  std::vector<std::shared_ptr<arrow::Array>> record_batches;
  ARROW_THROW_NOT_OK(ReadParquetFromS3(stream_fs.get(), config.stream_file,
                                       config.batch_size, record_batches, config.limit));
  sw.stop();
  json_download["stream_file_time"] = sw.ms();

  int64_t n_results = 0;
  int64_t n_points = 0;
  int64_t array_index_offset = 0;

  auto n_chunks = config.parallelism;
  ThreadPool pool(n_chunks);

  struct QueryResult {
    std::unique_ptr<std::vector<uint32_t>> build_indices;
    std::unique_ptr<std::vector<uint32_t>> stream_indices;
    nlohmann::json json_query;
    QueryResult() {
      build_indices = std::make_unique<std::vector<uint32_t>>();
      stream_indices = std::make_unique<std::vector<uint32_t>>();
    }
  };
  std::vector<std::future<QueryResult>> fut_results;

  sw.start();
  for (size_t i = 0; i < record_batches.size(); i++) {
    auto fn = [&](size_t batch_idx, int64_t array_index_offset) -> QueryResult {
      nanoarrow::UniqueArray unique_array;
      nanoarrow::UniqueSchema unique_schema;
      auto ctx = joiner->CreateContext();
      auto& array = record_batches[batch_idx];
      Stopwatch sw;

      ARROW_THROW_NOT_OK(
          arrow::ExportArray(*array, unique_array.get(), unique_schema.get()));
      QueryResult query_result;
      sw.start();
      joiner->PushStream(ctx.get(), &tmp_schema, unique_array.get(), 0, array->length(),
                         Predicate::kContains, query_result.build_indices.get(),
                         query_result.stream_indices.get(), array_index_offset);
      sw.stop();
      query_result.json_query["time"] = sw.ms();
      query_result.json_query["record_batch_idx"] = batch_idx;
      query_result.json_query["query_size"] = unique_array->length;
      query_result.json_query["result_size"] = query_result.build_indices->size();
      return query_result;
    };
    // fn(i, array_index_offset);
    fut_results.emplace_back(pool.enqueue(fn, i, array_index_offset));
    array_index_offset += record_batches[i]->length();
  }

  std::vector<QueryResult> results;

  for (auto& fut : fut_results) {
    results.emplace_back(fut.get());
  }
  sw.stop();
  query_ms = sw.ms();

  for (auto& res : results) {
    json_query.push_back(res.json_query);
    n_points += res.json_query.at("query_size").get<int64_t>();
    n_results += res.json_query.at("result_size").get<int64_t>();

    if (!config.output_file.empty()) {
      append_vector_pairs_to_file(config.output_file, *res.build_indices,
                                  *res.stream_indices);
    }
  }

  json["total_build_size"] = build_size;
  json["total_queries"] = n_points;
  json["total_results"] = n_results;
  json["total_query_time"] = query_ms;
  printf("Number of points %ld\n", n_points);
  printf("Total # of results: %ld\n", n_results);
  printf(" Query %lf ms\n", query_ms);
  ARROW_THROW_NOT_OK(arrow::fs::FinalizeS3());
  if (!config.json_file.empty()) {
    std::ofstream out(config.json_file);
    if (!out.is_open()) {
      throw std::runtime_error("Could not open json file " + config.json_file);
    }
    out << json.dump(4);
    out.close();
  }
}
}  // namespace gpuspatial