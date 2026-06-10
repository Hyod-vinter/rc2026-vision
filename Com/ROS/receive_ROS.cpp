#include "receive_ROS.hpp"

#pragma pack(push, 1)
struct RxPack
{
  double x;
  double y;
  uint16_t crc16;
};
#pragma pack(pop)

ReceiveROS::ReceiveROS(const rclcpp::Node::SharedPtr& node,
                       std::string topic,
                       rclcpp::QoS qos)
: node_(node)
{
  sub_ = node_->create_subscription<point_msgs::msg::SendROS>(
    topic, qos,
    std::bind(&ReceiveROS::cb, this, std::placeholders::_1)
  );

}

void ReceiveROS::cb(const point_msgs::msg::SendROS::SharedPtr msg)
{
  tools::logger()->debug("[ROS] x: {}, y: {}", msg->x, msg->y);
  
  RxPack pack{};
  pack.x = msg->x;
  pack.y = msg->y;

  // 更新 latest
  {
    std::lock_guard<std::mutex> lk(mtx_);
    latest_.x = pack.x;
    latest_.y = pack.y;
  }
}

ReceiveDataROS ReceiveROS::latest() const
{
  std::lock_guard<std::mutex> lk(mtx_);
  return latest_;
}

