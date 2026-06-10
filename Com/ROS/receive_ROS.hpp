#pragma once
#include <mutex>

#include <rclcpp/rclcpp.hpp>

#include "point_msgs/msg/send_ros.hpp"

#include "tools/crc.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/logger.hpp"



struct ReceiveDataROS
{
  float x;
  float y;
};

class ReceiveROS
{
public:
  ReceiveROS(const rclcpp::Node::SharedPtr& node,
             std::string topic = "send_ros",
             rclcpp::QoS qos = rclcpp::QoS(1));

  ReceiveDataROS latest() const;


private:
  void cb(const point_msgs::msg::SendROS::SharedPtr msg);

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<point_msgs::msg::SendROS>::SharedPtr sub_;

  // 最新数据
  mutable std::mutex mtx_;
  ReceiveDataROS latest_;

};
