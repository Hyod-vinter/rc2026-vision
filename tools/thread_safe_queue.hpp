#ifndef TOOLS__THREAD_SAFE_QUEUE_HPP
#define TOOLS__THREAD_SAFE_QUEUE_HPP

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <chrono>

namespace tools
{

template <typename T, bool PopWhenFull = false>
class ThreadSafeQueue
{
public:
  explicit ThreadSafeQueue(
    size_t max_size,
    std::function<void(void)> full_handler = [] {})
  : max_size_(max_size), full_handler_(full_handler)
  {
  }

  void push(const T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.size() >= max_size_) {
      if constexpr (PopWhenFull) {
        queue_.pop();   // 队列满了，丢旧帧
      } else {
        full_handler_();
        return;
      }
    }

    queue_.push(value);
    not_empty_condition_.notify_one();
  }

  void push(T && value)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.size() >= max_size_) {
      if constexpr (PopWhenFull) {
        queue_.pop();
      } else {
        full_handler_();
        return;
      }
    }

    queue_.push(std::move(value));
    not_empty_condition_.notify_one();
  }

  void pop(T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    not_empty_condition_.wait(lock, [this] {
      return !queue_.empty();
    });

    value = std::move(queue_.front());
    queue_.pop();
  }

  bool pop_for(T & value, std::chrono::milliseconds timeout)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    const bool ok = not_empty_condition_.wait_for(
      lock,
      timeout,
      [this] {
        return !queue_.empty();
      });

    if (!ok) {
      return false;
    }

    value = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  bool empty() const
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void clear()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
      queue_.pop();
    }
  }

private:
  std::queue<T> queue_;
  size_t max_size_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_condition_;
  std::function<void(void)> full_handler_;
};

}  // namespace tools

#endif
