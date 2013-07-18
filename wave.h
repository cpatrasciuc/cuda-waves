// Copyright (c) 2011 Cristian Patrasciuc. 
// Use of this source code is governed by an MIT license that can be
// found in the LICENSE file.

#ifndef WAVE_H
#define WAVE_H

#include <cmath>

struct Wave
{
    float wavelength;
    float amplitude;
    float speed;
    float dx;
    float dy;
    float phase;
};

inline float computeWave(const Wave *w, float x, float y, float t)
{
    // W(x,y,t) = A x sin(D * (x,y) x w + t x phi)
    float tmp = std::sin((w->dx * x + w->dy * y) / w->wavelength +
                         t * w->speed * 0.01 + w->phase) + 1;
    return w->amplitude * tmp * tmp / 2.0;
}

#endif // WAVE_H
