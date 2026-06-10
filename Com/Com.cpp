#include "Com.hpp"
#include "tools/crc.hpp"
#include "tools/yaml.hpp"


Com::Com(const std::string & config_path)
{
  auto yaml = tools::load(config_path);  

  auto port_ = tools::read<std::string>(yaml, "com_port"); 
  auto baudrate_ = tools::read<int>(yaml, "baudrate"); 
  
  try
  {
    serial_.setPort(port_);
    serial_.setBaudrate(baudrate_);
    serial::Timeout timeout = serial::Timeout::simpleTimeout(1000);
    serial_.setTimeout(timeout);
    serial_.open();
    serial_.flush();
  }
  catch (const std::exception & e) 
  {
    tools::logger()->error("[Com] Failed to open serial: {}", e.what());  
    exit(1);  
  }

  readthread_ = std::thread(&Com::read_thread, this);
}
Com::~Com()
{
    quit_ = true;
    try { serial_.close(); } catch (...) {}
    if (readthread_.joinable())
        readthread_.join();
}
WorklMode Com::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_); 
  return workl_mode_;
}
Receive_data Com::receive()
{
  std::lock_guard<std::mutex> lock(mutex_); 
  return receive_data_;
}
std::string Com::str(WorklMode mode) const
{
  switch (mode)  
  {
    case WorklMode::step1:
      return "step1";
    case WorklMode::step2:
      return "step2";
    case WorklMode::step3:
      return "step3";
    default:
      return "NONE"; 
  }
}

void Com::send(Send_data send_data)
{
    send_data_.workl_mode = send_data.workl_mode;
    send_data_.bool_graping = send_data.bool_graping;
    send_data_.bool_getKFS = send_data.bool_getKFS;
    send_data_.xyz_in_base[0] = send_data.xyz_in_base[0];
    send_data_.xyz_in_base[1] = send_data.xyz_in_base[1];
    send_data_.xyz_in_base[2] = send_data.xyz_in_base[2]; 
    send_data_.x_map = send_data.x_map;
    send_data_.y_map = send_data.y_map;
    send_data_.meilin_count = send_data.meilin_count;
    send_data_.meilin_points[0] = send_data.meilin_points[0];
    send_data_.meilin_points[1] = send_data.meilin_points[1];
    send_data_.meilin_points[2] = send_data.meilin_points[2];
    send_data_.meilin_points[3] = send_data.meilin_points[3];
    send_data_.meilin_points[4] = send_data.meilin_points[4];
    send_data_.meilin_points[5] = send_data.meilin_points[5];
    send_data_.meilin_points[6] = send_data.meilin_points[6];
    send_data_.meilin_points[7] = send_data.meilin_points[7];
    send_data_.meilin_points[8] = send_data.meilin_points[8];
    send_data_.i = send_data.i;
    //tools::logger()->debug("[Comsend] bool_graping: {}", static_cast<int>(send_data_.bool_graping));
  
    send_data_.crc16 = tools::get_crc16(reinterpret_cast<uint8_t *>(&send_data_), sizeof(send_data_) - sizeof(send_data_.crc16));

  try 
  {
    serial_.write(reinterpret_cast<uint8_t *>(&send_data_), sizeof(send_data_));
  } 
  catch (const std::exception & e)  // 捕获发送异常
  {
    tools::logger()->warn("[Com] Failed to write serial: {}", e.what());  // 打印警告日志
  }
}
bool Com::read(uint8_t * buffer, size_t size)
{
  try 
  {
    return serial_.read(buffer, size) == size;
  } 
  catch (const std::exception & e)  
  {
    tools::logger()->warn("[Com] Failed to read serial: {}", e.what());
    return false; 
  }
}
void Com::read_thread()
{
  tools::logger()->info("[Com] read_thread started.");  // 打印线程启动日志
  int error_count = 0;  // 错误计数器（用于触发重连）

  while (!quit_)  // 循环直到收到退出标志
  {
    // 若错误次数超过5000，尝试重连串口
    if (error_count > 5000) 
    {
      error_count = 0;  // 重置错误计数器
      tools::logger()->warn("[Com] Too many errors, attempting to reconnect...");  // 打印警告
      reconnect();  // 调用重连函数
      continue;  // 重连后继续循环
    }

    // 读取帧头（4字节）到接收缓冲区
    if (!read(reinterpret_cast<uint8_t *>(&receive_data_), sizeof(receive_data_.head))) 
    {
      error_count++;  
      continue;  
    }

    // 校验帧头是否为"MEIC"，否则跳过
    if (receive_data_.head[0] != 'M' || receive_data_.head[1] != 'E' || receive_data_.head[2] != 'I' || receive_data_.head[3] != 'C') 
      continue;

    auto t = std::chrono::steady_clock::now();  // 记录当前时间戳（数据接收时间）

    // 读取帧头之外的剩余数据（总大小 - 帧头大小）
    if (!read(reinterpret_cast<uint8_t *>(&receive_data_) + sizeof(receive_data_.head),sizeof(receive_data_) - sizeof(receive_data_.head))) 
    {
      error_count++;  
      continue;  
    }
    tools::logger()->debug("[Com] bool_graping: {}", static_cast<int>(receive_data_.bool_graping));

    if (!tools::check_crc16(reinterpret_cast<uint8_t *>(&receive_data_), sizeof(receive_data_))) 
    {
      tools::logger()->debug("[Com] CRC16 check failed.");
      error_count++;
      
      uint16_t expected_crc = tools::get_crc16(reinterpret_cast<uint8_t *>(&receive_data_), sizeof(receive_data_) - sizeof(receive_data_.crc16));
      
      uint16_t received_crc = receive_data_.crc16;
      tools::logger()->debug("[Com] Expected CRC: {:04X}, Received CRC: {:04X}", expected_crc, received_crc);
      
      continue;
    }

    error_count = 0;  // 数据读取和校验成功，重置错误计数器

    std::lock_guard<std::mutex> lock(mutex_);  // 加互斥锁（保护共享变量）

    // 根据接收的模式值更新云台模式枚举
    switch (receive_data_.workl_mode) 
    {
      case 0:
        workl_mode_ = WorklMode::step1;
        break;
      case 1:
        workl_mode_ = WorklMode::step2;
        //tools::logger()->warn("[Com]  workl_mode: {}", static_cast<int>(receive_data_.workl_mode));
        break;
      case 2:
        workl_mode_ = WorklMode::step3;
        break;
      default:
        workl_mode_ = WorklMode::NONE;  
        int invalid_workl_mode = static_cast<int>(receive_data_.workl_mode);
        tools::logger()->warn("[Com] Invalid workl_mode: {}", invalid_workl_mode);
        break;
    }

     //tools::logger()->debug("[Comreceive] bool_graping: {}", (bool)receive_data_.bool_graping);

    
  }

  tools::logger()->info("[Com] read_thread stopped.");  
}
void Com::reconnect()
{
  int max_retry_count = 10;  

  for (int i = 0; i < max_retry_count && !quit_; ++i) 
  {
    tools::logger()->warn("[Com] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);  // 打印重试日志
    try 
    {
      serial_.close();  
      std::this_thread::sleep_for(std::chrono::seconds(1));  
    } 
    catch (...)  
    {
    }
    try 
    {
      serial_.open();  
      tools::logger()->info("[Com] Reconnected serial successfully.");  
      break;  
    } catch (const std::exception & e)  
    {
      tools::logger()->warn("[Com] Reconnect failed: {}", e.what());  
      std::this_thread::sleep_for(std::chrono::seconds(1));  
    }
  }
}