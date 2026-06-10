#pragma once

#include <stdint.h>   // 用于 uint8_t, int16_t 等类型
#include <atomic>
#include <string>
#include <mutex>
#include <thread>
#include <tuple>
#include "serial/serial.h"
#include <yaml-cpp/yaml.h>


struct Receive_data {
    
    uint8_t head[4] = {'M', 'E', 'I', 'C'};
    int workl_mode;
    int bool_graping;
    int yaw;
    int pitch;
    uint16_t crc16;  // CRC16校验值（用于数据完整性校验）
 
}__attribute__((packed));

enum : uint8_t
{
    DIR_NONE  = 0,
    DIR_LEFT  = 1,
    DIR_RIGHT = 2,
    DIR_FRONT = 3,
    DIR_BACK  = 4,
    DIR_UP    = 5,
    DIR_DOWN  = 6
};

struct MeilinPointCmd
{
    uint8_t cell;        // 点位：1~12，0无效
    uint8_t horizontal_s;    // 左1 右2 前3 后4，0无
    uint8_t vertical_s;  // 上5 下6，0无
} __attribute__((packed));

struct Send_data {

    uint8_t head[4] = {'M', 'E', 'I', 'C'};
    int workl_mode;
    int bool_graping;
    int bool_getKFS;
    float xyz_in_base[3];
    //float wepend_offset_x;
    float x_map;
    float y_map;
    //float yaw_map;
    int i=1;
    uint8_t meilin_count = 0;               // 有效点数量，最多9
    MeilinPointCmd meilin_points[9] = {};   // 不足9个填{0,0,0}
    uint16_t crc16;  // CRC16校验值（用于数据完整性校验）
  
}__attribute__((packed));

enum class WorklMode
{
  step1,
  step2,
  step3,
  NONE
};

class Com {
public:
    Com(const std::string & config_path);
    ~Com();
    WorklMode mode() const;
    std::string str(WorklMode mode) const;
    void send(Send_data send_data);
    Receive_data receive();
private:
    serial::Serial serial_; 
    std::thread readthread_;
    std::atomic<bool> quit_{false};
    mutable std::mutex mutex_;
    Receive_data receive_data_;
    Send_data send_data_;
    WorklMode workl_mode_{WorklMode::NONE};

    bool read(uint8_t * buffer, size_t size);
    void read_thread();
    void reconnect();

};
