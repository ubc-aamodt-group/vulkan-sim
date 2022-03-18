#ifndef INTEL_IMAGE_UTIL_H
#define INTEL_IMAGE_UTIL_H

#include <fstream>
#include <iostream>
#include <assert.h>

#define HAVE_PTHREAD
#define UTIL_ARCH_LITTLE_ENDIAN 1
#define UTIL_ARCH_BIG_ENDIAN 0
#define signbit signbit

#define UINT_MAX 65535
#define GLuint MESA_GLuint
#include "isl/isl.h"
#include "vulkan/anv_private.h"
#undef GLuint

#include "vulkan/anv_public.h"
#include "astc_decomp.h"


void save_ASTC_texture_to_image_file(const struct anv_image *image, std::ofstream &imageFile)
{
    assert(image->vk_format == VK_FORMAT_ASTC_8x8_SRGB_BLOCK);
    uint8_t* address = (uint8_t*)anv_address_map(image->planes[0].address);

    printf("writing image with extant = (%d, %d)\n", image->extent.height, image->extent.width);

    const struct anv_format_plane* plane_format = &image->format->planes[0];

    const struct isl_surf* isl_surface = &image->planes[0].surface.isl;

    uint32_t tileWidth = 8;
    uint32_t tileHeight = 32;
    uint32_t ASTC_block_size = 128 / 8;
    
    for(int x = 0; x < image->extent.width; x++)
    {
        for(int y = 0; y < image->extent.height; y++)
        {
            int tileX = x / 8 / tileWidth;
            int tileY = y / 8 / tileHeight;
            int tileID = tileX + tileY * image->extent.width / 8 / tileWidth;

            int blockX = ((x / 8) % tileWidth);
            int blockY = ((y / 8) % tileHeight);
            int blockID = blockX * (tileHeight) + blockY;

            uint32_t offset = (tileID * (tileWidth * tileHeight) + blockID) * ASTC_block_size;
            // uint32_t offset = (blockX + blockY * (image->extent.width / 8)) * (128 / 8);
            // uint32_t offset = (blockX * (image->extent.height / 8) + blockY) * (128 / 8);

            uint8_t dst_colors[100];
            if(!basisu::astc::decompress(dst_colors, address + offset, true, 8, 8))
            {
                printf("decoding error\n");
                exit(-2);
            }
            uint8_t* pixel_color = &dst_colors[0] + ((x % 8) + (y % 8) * 8) * 4;

            uint32_t bit_map_offset = x + y * image->extent.width;

            float data[4];
            data[0] = pixel_color[0] / 255.0;
            data[1] = pixel_color[1] / 255.0;
            data[2] = pixel_color[2] / 255.0;
            data[3] = pixel_color[3] / 255.0;
            imageFile.write((char*) data, 3 * sizeof(float));
            imageFile.write((char*) (&bit_map_offset), sizeof(uint32_t));
            imageFile.flush();
        }
    }
}



// uint32_t get_ASTC_block_offset(const struct anv_image *image, uint32_t x, uint32_t y)
// {
//     switch (image.)
//     {
//     case /* constant-expression */:
//         /* code */
//         break;
    
//     default:
//         break;
//     }
// }

// float4 read_pixel(const struct anv_image *image, uint32_t x, uint32_t y)
// {
//     assert(image->samples == 1);
//     assert(image->array_size == 1);
//     assert(image->n_planes == 1);

//     const struct isl_surf* isl_surface = &image->planes[0].surface.isl;
// }






#endif /* INTEL_IMAGE_UTIL_H */