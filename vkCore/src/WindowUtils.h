#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <cassert>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <spdlog/spdlog.h>

static const unsigned int WIDTH = 123 * 5;
static const unsigned int HEIGHT = 135 * 5;

using namespace std;
namespace WindowUtil
{
    struct Rect {
        Rect()
            : left(0)
            , bottom(0)
            , width(0)
            , height(0) {}
        Rect(int l, int b, int w, int h)
            : left(l)
            , bottom(b)
            , width(w)
            , height(h) {}
        int left;
        int bottom;
        int width;
        int height;

        bool pointInRect(int x, int y) {
            if (x >= left && x <= left + width && y >= bottom && y <= bottom + height)
                return true;
            return false;
        }

        bool isEuqal(const Rect& r) {
            if (left == r.left && bottom == r.bottom && width == r.width && height == r.height)
                return true;
            else
                return false;
        }
    };
}  // namespace WindowUtil