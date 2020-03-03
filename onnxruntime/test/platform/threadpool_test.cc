// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/threadpool.h"
#include "core/platform/EigenNonBlockingThreadPool.h"

#include <core/common/make_unique.h>

#include "gtest/gtest.h"
#include <algorithm>
#include <memory>
#include <functional>
#include <mutex>

using namespace onnxruntime::concurrency;

namespace {

struct TestData {
  explicit TestData(int num) : data(num, 0) {
  }
  std::vector<int> data;
  std::mutex mutex;
};

// This unittest tests ThreadPool function by counting the number of calls to function with each index.
// the function should be called exactly once for each element.

std::unique_ptr<TestData> CreateTestData(int num) {
  return onnxruntime::make_unique<TestData>(num);
}

void IncrementElement(TestData& test_data, ptrdiff_t i) {
  std::lock_guard<std::mutex> lock(test_data.mutex);
  test_data.data[i]++;
}

void ValidateTestData(TestData& test_data) {
  ASSERT_TRUE(std::count_if(test_data.data.cbegin(), test_data.data.cend(), [](int i) { return i != 1; }) == 0);
}

void CreateThreadPoolAndTest(const std::string&, int num_threads, const std::function<void(ThreadPool*)>& test_body) {
  auto tp = onnxruntime::make_unique<ThreadPool>(&onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr,
                                                 num_threads, true);
  test_body(tp.get());
}

void TestParallelFor(const std::string& name, int num_threads, int num_tasks) {
  auto test_data = CreateTestData(num_tasks);
  CreateThreadPoolAndTest(name, num_threads, [&](ThreadPool* tp) {
    tp->ParallelFor(num_tasks, [&](int i) { IncrementElement(*test_data, i); });
  });
  ValidateTestData(*test_data);
}

void TestBatchParallelFor(const std::string& name, int num_threads, int num_tasks, int batch_size) {
  auto test_data = CreateTestData(num_tasks);

  CreateThreadPoolAndTest(name, num_threads, [&](ThreadPool* tp) {
    onnxruntime::concurrency::ThreadPool::TryBatchParallelFor(
        tp, num_tasks, [&](ptrdiff_t i) { IncrementElement(*test_data, i); }, batch_size);
  });
  ValidateTestData(*test_data);
}

}  // namespace

namespace onnxruntime {
TEST(ThreadPoolTest, TestParallelFor_2_Thread_NoTask) {
  TestParallelFor("TestParallelFor_2_Thread_NoTask", 2, 0);
}

TEST(ThreadPoolTest, TestParallelFor_2_Thread_50_Task) {
  TestParallelFor("TestParallelFor_2_Thread_50_Task", 2, 50);
}

TEST(ThreadPoolTest, TestParallelFor_1_Thread_50_Task) {
  TestParallelFor("TestParallelFor_1_Thread_50_Task", 1, 50);
}

TEST(ThreadPoolTest, TestBatchParallelFor_2_Thread_50_Task_10_Batch) {
  TestBatchParallelFor("TestBatchParallelFor_2_Thread_50_Task_10_Batch", 2, 50, 10);
}

TEST(ThreadPoolTest, TestBatchParallelFor_2_Thread_50_Task_0_Batch) {
  TestBatchParallelFor("TestBatchParallelFor_2_Thread_50_Task_0_Batch", 2, 50, 0);
}

TEST(ThreadPoolTest, TestBatchParallelFor_2_Thread_50_Task_1_Batch) {
  TestBatchParallelFor("TestBatchParallelFor_2_Thread_50_Task_1_Batch", 2, 50, 1);
}

TEST(ThreadPoolTest, TestBatchParallelFor_2_Thread_50_Task_100_Batch) {
  TestBatchParallelFor("TestBatchParallelFor_2_Thread_50_Task_100_Batch", 2, 50, 100);
}

TEST(ThreadPoolTest, TestBatchParallelFor_2_Thread_81_Task_20_Batch) {
  TestBatchParallelFor("TestBatchParallelFor_2_Thread_81_Task_20_Batch", 2, 81, 20);
}

// Sadly, Eigen threadpool doesn't support nested parallelFor. Java can do it, C# can do it, TBB can do it,
// but not Eigen.
// TEST(ThreadPoolTest, Nested) {
//  const int num_threads = 10;
//  ThreadPool tp(&Env::Default(), ThreadOptions(), "", num_threads, true, nullptr);
//  onnxruntime::Notification finished;
//  tp.AsEigenThreadPool()->Schedule([&finished, &tp, num_threads]() {
//    onnxruntime::Barrier b(num_threads);
//    tp.device().parallelFor(
//        num_threads,
//        Eigen::TensorOpCost(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
//                            std::numeric_limits<float>::max()),
//        [num_threads, &b, &tp](Eigen::Index start, Eigen::Index end) {
//          ASSERT_EQ(start + 1, end);
//          b.Notify();
//          b.Wait();
//          tp.device().parallelFor(
//              num_threads,
//              Eigen::TensorOpCost(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
//                                  std::numeric_limits<float>::max()),
//              [&tp](Eigen::Index, Eigen::Index) {
//                  auto id = tp.AsEigenThreadPool()->CurrentThreadId();
//                  std::cout << "Thread "<<id << ": test output from nested loop" << std::endl; });
//        });
//    finished.Notify();
//  });
//  finished.Wait();
//}
//
// TEST(ThreadPoolTest, Nested2) {
//  const int num_threads = 10;
//  ThreadPool tp(&Env::Default(), ThreadOptions(), "", num_threads, true, nullptr);
//  onnxruntime::Notification finished;
//  tp.AsEigenThreadPool()->Schedule([&finished, &tp, num_threads]() {
//    onnxruntime::Barrier b(num_threads);
//    concurrency::ThreadPool::SchedulingParams sp(concurrency::ThreadPool::SchedulingStrategy::kFixedBlockSize,
//                                                 optional<int64_t>(), optional<int64_t>(1));
//
//    tp.ParallelFor(num_threads, sp, [num_threads, &b, &tp, sp](Eigen::Index start, Eigen::Index end) {
//      ASSERT_EQ(start + 1, end);
//      b.Notify();
//      b.Wait();
//      tp.ParallelFor(num_threads, sp,
//                     [&tp](Eigen::Index, Eigen::Index) {
//              auto id = tp.AsEigenThreadPool()->CurrentThreadId();
//        std::cout << "Thread " << id << ": test output from nested loop" << std::endl;
//      });
//    });
//  });
//  finished.Wait();
//}

}  // namespace onnxruntime