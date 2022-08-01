// Copyright (c) 2022, Mohammadreza Saed, Yuan Hsi Chou, Lufei Liu, Tor M. Aamodt,
// The University of British Columbia
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ANV_INCLUDE_H
#define ANV_INCLUDE_H

#define HAVE_PTHREAD
#define UTIL_ARCH_LITTLE_ENDIAN 1
#define UTIL_ARCH_BIG_ENDIAN 0
#define signbit signbit

#define UINT_MAX 65535
#define GLuint MESA_GLuint
#include "isl/isl.h"
#include "isl/isl_tiled_memcpy.c"
#include "vulkan/anv_private.h"
#undef GLuint

#undef HAVE_PTHREAD
#undef UTIL_ARCH_LITTLE_ENDIAN
#undef UTIL_ARCH_BIG_ENDIAN
#undef signbit

#include "vulkan/anv_public.h"

#endif