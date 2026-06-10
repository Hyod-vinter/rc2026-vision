#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>

#include "realsense/realsense.hpp"

struct GridResult {
    cv::Mat warped;
    std::vector<cv::Rect> cells;
    std::vector<int> xs;
    std::vector<int> ys;
};

static std::vector<int> findLineCentersByProjection(
    const cv::Mat& binary,
    bool vertical,
    int min_peak_width = 8
) {
    std::vector<int> centers;
    cv::Mat proj;

    if (vertical) {
        cv::reduce(binary, proj, 0, cv::REDUCE_SUM, CV_32S);
    } else {
        cv::reduce(binary, proj, 1, cv::REDUCE_SUM, CV_32S);
    }

    int len = vertical ? binary.cols : binary.rows;
    std::vector<int> values(len);

    for (int i = 0; i < len; i++) {
        values[i] = vertical ? proj.at<int>(0, i) : proj.at<int>(i, 0);
        values[i] /= 255;
    }

    int maxv = *std::max_element(values.begin(), values.end());
    int thresh = std::max(20, maxv / 3);

    bool in_peak = false;
    int start = 0;

    for (int i = 0; i < len; i++) {
        if (values[i] > thresh && !in_peak) {
            start = i;
            in_peak = true;
        }

        if ((values[i] <= thresh || i == len - 1) && in_peak) {
            int end = i;

            if (end - start >= min_peak_width) {
                centers.push_back((start + end) / 2);
            }

            in_peak = false;
        }
    }

    return centers;
}

static void keepFourLines(std::vector<int>& lines, int total_len)
{
    std::sort(lines.begin(), lines.end());

    if (lines.size() <= 4) return;

    std::vector<int> targets = {
        0,
        total_len / 3,
        total_len * 2 / 3,
        total_len - 1
    };

    std::vector<int> picked;

    for (int t : targets) {
        int best = lines[0];
        int best_dist = std::abs(lines[0] - t);

        for (int x : lines) {
            int d = std::abs(x - t);
            if (d < best_dist) {
                best = x;
                best_dist = d;
            }
        }

        picked.push_back(best);
    }

    std::sort(picked.begin(), picked.end());
    picked.erase(std::unique(picked.begin(), picked.end()), picked.end());

    lines = picked;
}

GridResult detectGridTraditional(
    const cv::Mat& color_bgr,
    const std::vector<cv::Point2f>& corners,
    int warp_w = 900,
    int warp_h = 900
) {
    GridResult result;

    std::vector<cv::Point2f> dst = {
        cv::Point2f(0, 0),
        cv::Point2f(warp_w - 1, 0),
        cv::Point2f(warp_w - 1, warp_h - 1),
        cv::Point2f(0, warp_h - 1)
    };

    cv::Mat H = cv::getPerspectiveTransform(corners, dst);
    cv::warpPerspective(color_bgr, result.warped, H, cv::Size(warp_w, warp_h));

    cv::Mat gray;
    cv::cvtColor(result.warped, gray, cv::COLOR_BGR2GRAY);

    // 检测黑胶带
    cv::Mat black_mask;
    cv::threshold(gray, black_mask, 85, 255, cv::THRESH_BINARY_INV);

    cv::morphologyEx(
        black_mask,
        black_mask,
        cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3))
    );

    cv::Mat vertical_mask;
    cv::morphologyEx(
        black_mask,
        vertical_mask,
        cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 90))
    );

    cv::Mat horizontal_mask;
    cv::morphologyEx(
        black_mask,
        horizontal_mask,
        cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(90, 7))
    );

    result.xs = findLineCentersByProjection(vertical_mask, true);
    result.ys = findLineCentersByProjection(horizontal_mask, false);

    keepFourLines(result.xs, warp_w);
    keepFourLines(result.ys, warp_h);

    if (result.xs.size() != 4 || result.ys.size() != 4) {
        std::cerr << "九宫格检测失败 xs=" << result.xs.size()
                  << " ys=" << result.ys.size() << std::endl;
        return result;
    }

    int margin = 10;

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            int x1 = result.xs[c];
            int x2 = result.xs[c + 1];
            int y1 = result.ys[r];
            int y2 = result.ys[r + 1];

            result.cells.emplace_back(
                x1 + margin,
                y1 + margin,
                std::max(1, x2 - x1 - 2 * margin),
                std::max(1, y2 - y1 - 2 * margin)
            );
        }
    }

    return result;
}

int main()
{
    camera::RealSense::StreamProfile prof;
    camera::RealSense rs("8086:0b3a", prof);

    while (!rs.is_ready()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::cout << rs.get_camera_info_string() << std::endl;

    cv::Mat color_bgr;
    cv::Mat depth_u16;

    while (true) {
        std::chrono::steady_clock::time_point ts;
        rs.read(color_bgr, depth_u16, ts);

        if (color_bgr.empty() || depth_u16.empty()) {
            continue;
        }

        // 这里先写死四个角点，后续可以换成 YOLO/关键点检测结果
        // 顺序：左上、右上、右下、左下
        std::vector<cv::Point2f> corners = {
            cv::Point2f(66, 57),
            cv::Point2f(817, 40),
            cv::Point2f(817, 790),
            cv::Point2f(66, 790)
        };

        GridResult grid = detectGridTraditional(color_bgr, corners);

        cv::Mat show = color_bgr.clone();

        for (auto& p : corners) {
            cv::circle(show, p, 5, cv::Scalar(0, 255, 0), -1);
        }

        cv::imshow("RealSense Color", show);

        if (!grid.warped.empty()) {
            cv::Mat vis = grid.warped.clone();

            for (int i = 0; i < grid.cells.size(); i++) {
                cv::rectangle(vis, grid.cells[i], cv::Scalar(0, 255, 0), 2);

                cv::putText(
                    vis,
                    std::to_string(i),
                    grid.cells[i].tl() + cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 0, 255),
                    2
                );
            }

            cv::imshow("Grid Cells", vis);
        }

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}

