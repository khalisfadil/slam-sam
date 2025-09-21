#pragma once

#include <iostream>
#include <thread>
#include <chrono>
#include <cstdint>

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <lidarcallback.hpp>
#include <compcallback.hpp>
#include <registercallback.hpp>
#include <udpsocket.hpp>
#include <dataframe.hpp>
#include <map.hpp>

using gtsam::Symbol;

template<typename T>
class FrameQueue {
    public:
        void push(std::unique_ptr<T> frame) {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(frame));
            cv_.notify_one();
        }
        std::unique_ptr<T> pop() {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });
            if (queue_.empty() && stopped_) return nullptr;
            std::unique_ptr<T> frame = std::move(queue_.front());
            queue_.pop();
            return frame;
        }
        void stop() {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
            cv_.notify_all();
        }
        size_t size() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }
    private:
        std::queue<std::unique_ptr<T>> queue_;
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        bool stopped_ = false;
};