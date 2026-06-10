#include <array>
#include <list>
#include <vector>
#include <algorithm>
#include <cmath>
#include <optional>

class WependTargetSelector
{
public:
  // targetid:       1  2  3  4  5  6
  // expected class: 矛 拳 掌 掌 拳 矛
  static constexpr std::array<int, 6> kSlotClass = {0, 1, 2, 2, 1, 0};
  static constexpr std::array<int, 6> kPickOrder = {2, 5, 3, 4, 1, 6};

  int update(const std::list<detector::WependObject> & objs)
  {
    // 有 6 个完整目标时，自动标定槽位 x 坐标
    try_calibrate(objs);

    if (!calibrated_) {
      return 0;  // 还没标定成功，先不发
    }

    std::array<bool, 6> now_visible{};
    now_visible.fill(false);

    for (const auto & obj : objs) {
      int slot = match_slot(obj);
      if (slot >= 0) {
        now_visible[slot] = true;
      }
    }

    // 防抖：连续识别到才认为存在，短暂漏检不马上认为被拿走
    for (int i = 0; i < 6; ++i) {
      if (now_visible[i]) {
        stable_score_[i] = std::min(stable_score_[i] + 1, 5);
      } else {
        stable_score_[i] = std::max(stable_score_[i] - 1, 0);
      }
    }

    // 按优先级找 targetid
    for (int targetid : kPickOrder) {
      int idx = targetid - 1;
      if (stable_score_[idx] >= 2) {
        return targetid;
      }
    }

    return 0;  // 没有可取目标
  }

private:
  bool calibrated_{false};
  float pitch_{0.0f};
  std::array<float, 6> slot_x_{};
  std::array<int, 6> stable_score_{};

  void try_calibrate(const std::list<detector::WependObject> & objs)
  {
    if (objs.size() < 6) {
      return;
    }

    std::vector<detector::WependObject> v(objs.begin(), objs.end());

    std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) {
      return a.center.x < b.center.x;
    });

    if (v.size() != 6) {
      return;
    }

    // 必须满足从左到右：矛 拳 掌 掌 拳 矛
    for (int i = 0; i < 6; ++i) {
      if (v[i].class_id != kSlotClass[i]) {
        return;
      }
    }

    for (int i = 0; i < 6; ++i) {
      slot_x_[i] = v[i].center.x;
    }

    pitch_ = (slot_x_[5] - slot_x_[0]) / 5.0f;

    if (pitch_ <= 1.0f) {
      return;
    }

    calibrated_ = true;

    // 标定成功时，默认 6 个都存在
    for (int i = 0; i < 6; ++i) {
      stable_score_[i] = 5;
    }

    //std::cout << "[WEPEND] calibrated, pitch=" << pitch_ << std::endl;
  }

  int match_slot(const detector::WependObject & obj) const
  {
    int best_slot = -1;
    float best_dist = 1e9f;

    for (int i = 0; i < 6; ++i) {
      if (obj.class_id != kSlotClass[i]) {
        continue;
      }

      float dist = std::abs(obj.center.x - slot_x_[i]);

      if (dist < best_dist) {
        best_dist = dist;
        best_slot = i;
      }
    }

    // 阈值不要太大，防止掌 3 / 掌 4 互相误匹配
    const float max_dist = pitch_ * 0.45f;

    if (best_slot >= 0 && best_dist < max_dist) {
      return best_slot;
    }

    return -1;
  }
};

