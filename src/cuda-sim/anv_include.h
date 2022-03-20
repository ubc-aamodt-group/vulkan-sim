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