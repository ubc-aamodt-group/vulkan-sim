#ifndef INTEL_IMAGE_H
#define INTEL_IMAGE_H

#include <fstream>
#include <vector>
#include <iostream>
#include <assert.h>
#include "astc_decomp.h"

#include "anv_include.h"

typedef struct Pixel{
    Pixel(float c0, float c1, float c2, float c3)
    : c0(c0), c1(c1), c2(c2), c3(c3) {}
    Pixel() {}

    union
    {
        float r;
        float c0;
    };
    union
    {
        float g;
        float c1;
    };
    union
    {
        float b;
        float c2;
    };
    union
    {
        float a;
        float c3;
    };
} Pixel;

enum class ImageTransactionType {
    TEXTURE_LOAD,
    IMAGE_LOAD,
    IMAGE_STORE,
};

typedef struct ImageMemoryTransactionRecord {
    ImageMemoryTransactionRecord(void* address, uint32_t size, ImageTransactionType type)
    : address(address), size(size), type(type) {}
    ImageMemoryTransactionRecord() {}
    void* address;
    uint32_t size;
    ImageTransactionType type;
} ImageMemoryTransactionRecord;

float SRGB_to_linearRGB(float s)
{
    assert(0 <= s && s <= 1);
    if(s <= 0.04045)
        return s / 12.92;
    else
        return pow(((s + 0.055) / 1.055), 2.4);
}

float linearRGB_to_SRGB(float s)
{
    // assert(0 <= s && s <= 1);
    if(s < 0.0031308)
        return s * 12.92;
    else
        return 1.055 * pow(s, 1 / 2.4) - 0.055;
}

Pixel load_image_pixel(const struct anv_image *image, uint32_t x, uint32_t y, uint32_t level, ImageMemoryTransactionRecord& transaction)
{
    assert(image->n_planes == 1);
    assert(image->samples == 1);
    assert(image->tiling == VK_IMAGE_TILING_OPTIMAL);
    assert(image->planes[0].surface.isl.tiling == ISL_TILING_Y0);
    assert(level == 0);

    uint8_t* address = anv_address_map(image->planes[0].address);

    switch(image->vk_format)
    {
        case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:
        {
            uint32_t tileWidth = 8;
            uint32_t tileHeight = 32;
            uint32_t ASTC_block_size = 128 / 8;

            int tileX = x / 8 / tileWidth;
            int tileY = y / 8 / tileHeight;
            int tileID = tileX + tileY * image->extent.width / 8 / tileWidth;

            int blockX = ((x / 8) % tileWidth);
            int blockY = ((y / 8) % tileHeight);
            int blockID = blockX * (tileHeight) + blockY;

            uint32_t offset = (tileID * (tileWidth * tileHeight) + blockID) * ASTC_block_size;
            
            transaction.address = address + offset;
            transaction.size = 128 / 8;

            uint8_t dst_colors[256];
            if(!basisu::astc::decompress(dst_colors, address + offset, true, 8, 8))
            {
                printf("decoding error\n");
                exit(-2);
            }
            uint8_t* pixel_color = &dst_colors[0] + ((x % 8) + (y % 8) * 8) * 4;

            Pixel pixel;
            pixel.r = SRGB_to_linearRGB(pixel_color[0] / 255.0);
            pixel.g = SRGB_to_linearRGB(pixel_color[1] / 255.0);
            pixel.b = SRGB_to_linearRGB(pixel_color[2] / 255.0);
            pixel.a = pixel_color[3] / 255.0;
            return pixel;
        }
        case VK_FORMAT_R8G8B8A8_SRGB:
        {
            uint32_t tileWidth = 32;
            uint32_t tileHeight = 32;
            int tileX = x / tileWidth;
            int tileY = y / tileHeight;
            int tileID = tileX + tileY * image->extent.width / tileWidth;

            transaction.address = address + (tileID * tileWidth * tileHeight + (x % tileWidth) * tileHeight + (y % tileHeight)) * 4;
            transaction.size = 4;

            uint8_t colors[4];

            intel_tiled_to_linear(x * 4, x * 4 + 4, y, y + 1,
                colors, address, image->extent.width * 4 ,image->planes[0].surface.isl.row_pitch_B, false,
                ISL_TILING_Y0, ISL_MEMCPY);

            Pixel pixel;
            pixel.r = SRGB_to_linearRGB(colors[0] / 255.0);
            pixel.g = SRGB_to_linearRGB(colors[1] / 255.0);
            pixel.b = SRGB_to_linearRGB(colors[2] / 255.0);
            pixel.a = colors[3] / 255.0;
            return pixel;
        }
        default:
        {
            printf("%d not implemented\n", image->vk_format);
            assert(0);
            break;
        }
    }
}

Pixel get_interpolated_pixel(struct anv_image_view *image_view, struct anv_sampler *sampler, float x, float y, std::vector<ImageMemoryTransactionRecord>& transactions)
{
    const struct anv_image *image = image_view->image;
    assert(sampler->conversion == NULL);

    VkFilter filter;
    if(sampler->conversion == NULL)
        filter = VK_FILTER_NEAREST;
    
    switch (filter)
    {
        case VK_FILTER_NEAREST:
        {
            uint32_t x_int = x * image->extent.width; //MRS_TODO: change this to NN or bilinear
            x_int %= image->extent.width;
            if(x_int < 0)
                x_int += image->extent.width;
            uint32_t y_int = y * image->extent.height;
            y_int %= image->extent.height;
            if(y_int < 0)
                y_int += image->extent.height;
            ImageMemoryTransactionRecord transaction;
            Pixel pixel = load_image_pixel(image, x_int, y_int, 0, transaction);
            transactions.push_back(transaction);
            return pixel;
        }
        default:
        {
            assert(0);
            break;
        }
    }
}

inline uint64_t ceil_divide(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

void store_image_pixel(const struct anv_image *image, uint32_t x, uint32_t y, uint32_t level, Pixel pixel, ImageMemoryTransactionRecord& transaction)
{
    assert(image->n_planes == 1);
    assert(image->samples == 1);
    assert(level == 0);

    void* address = anv_address_map(image->planes[0].address);

    switch (image->vk_format)
    {
        case VK_FORMAT_B8G8R8A8_UNORM:
        {
            uint8_t r = pixel.r * 255;
            uint8_t g = pixel.g * 255;
            uint8_t b = pixel.b * 255;
            uint8_t a = pixel.a * 255;

            if(pixel.r >= 1)
                r = 255;
            if(pixel.g >= 1)
                g = 255;
            if(pixel.b >= 1)
                b = 255;
            if(pixel.a >= 1)
                a = 255;
            
            uint8_t colors[] = {r, g, b, a};

            switch (image->planes[0].surface.isl.tiling)
            {
                case ISL_TILING_Y0:
                {
                    uint32_t tileWidth = 32;
                    uint32_t tileHeight = 32;
                    int tileX = x / tileWidth;
                    int tileY = y / tileHeight;
                    int tileID = tileX + tileY * ceil_divide(image->extent.width, tileWidth);

                    uint32_t offset = (tileID * tileWidth * tileHeight + (x % tileWidth) + (y % tileHeight) * tileWidth) * 4;
                    transaction.address = address + offset;
                    transaction.size = 4;

                    assert(image->tiling == VK_IMAGE_TILING_OPTIMAL);
                    intel_linear_to_tiled(x * 4, x * 4 + 4, y, y + 1,
                        (char *)address, colors, image->planes[0].surface.isl.row_pitch_B, 1280 * 4, false,
                        ISL_TILING_Y0, ISL_MEMCPY_BGRA8);
                    break;
                }
            
            default:
                assert(0);
                break;
            }

            break;
        }
        case VK_FORMAT_R8G8B8A8_UNORM:
        {
            uint8_t r = pixel.r * 255;
            uint8_t g = pixel.g * 255;
            uint8_t b = pixel.b * 255;
            uint8_t a = pixel.a * 255;

            if(pixel.r >= 1)
                r = 255;
            if(pixel.g >= 1)
                g = 255;
            if(pixel.b >= 1)
                b = 255;
            if(pixel.a >= 1)
                a = 255;
            
            uint8_t colors[] = {r, g, b, a};

            switch (image->planes[0].surface.isl.tiling)
            {
                case ISL_TILING_Y0:
                {
                    uint32_t tileWidth = 32;
                    uint32_t tileHeight = 32;
                    int tileX = x / tileWidth;
                    int tileY = y / tileHeight;
                    int tileID = tileX + tileY * ceil_divide(image->extent.width, tileWidth);

                    uint32_t offset = (tileID * tileWidth * tileHeight + (x % tileWidth) + (y % tileHeight) * tileWidth) * 4;
                    transaction.address = address + offset;
                    transaction.size = 4;

                    assert(image->tiling == VK_IMAGE_TILING_OPTIMAL);
                    intel_linear_to_tiled(x * 4, x * 4 + 4, y, y + 1,
                        (char *)address, colors, image->planes[0].surface.isl.row_pitch_B, 1280 * 4, false,
                        ISL_TILING_Y0, ISL_MEMCPY_BGRA8);
                    break;
                }

                case ISL_TILING_LINEAR:
                {
                    uint32_t offset = (y * image->extent.width + x) * 4;

                    transaction.address = address + offset;
                    transaction.size = 4;

                    uint8_t* p = ((uint8_t*)address) + offset;
                    p[0] = r;
                    p[1] = g;
                    p[2] = b;
                    p[3] = a;
                    break;
                }
            
                default:
                    assert(0);
                    break;
            }
            break;
        }
        
        default:
            assert(0);
            break;
    }
}

// void save_ASTC_texture_to_image_file(const struct anv_image *image, std::ofstream &imageFile)
// {
//     assert(image->vk_format == VK_FORMAT_ASTC_8x8_SRGB_BLOCK);
//     uint8_t* address = (uint8_t*)anv_address_map(image->planes[0].address);

//     printf("writing image with extant = (%d, %d)\n", image->extent.height, image->extent.width);

//     const struct anv_format_plane* plane_format = &image->format->planes[0];

//     const struct isl_surf* isl_surface = &image->planes[0].surface.isl;

//     uint32_t tileWidth = 8;
//     uint32_t tileHeight = 32;
//     uint32_t ASTC_block_size = 128 / 8;
    
//     for(int x = 0; x < image->extent.width; x++)
//     {
//         for(int y = 0; y < image->extent.height; y++)
//         {
//             int tileX = x / 8 / tileWidth;
//             int tileY = y / 8 / tileHeight;
//             int tileID = tileX + tileY * image->extent.width / 8 / tileWidth;

//             int blockX = ((x / 8) % tileWidth);
//             int blockY = ((y / 8) % tileHeight);
//             int blockID = blockX * (tileHeight) + blockY;

//             uint32_t offset = (tileID * (tileWidth * tileHeight) + blockID) * ASTC_block_size;
//             // uint32_t offset = (blockX + blockY * (image->extent.width / 8)) * (128 / 8);
//             // uint32_t offset = (blockX * (image->extent.height / 8) + blockY) * (128 / 8);

//             uint8_t dst_colors[100];
//             if(!basisu::astc::decompress(dst_colors, address + offset, true, 8, 8))
//             {
//                 printf("decoding error\n");
//                 exit(-2);
//             }
//             uint8_t* pixel_color = &dst_colors[0] + ((x % 8) + (y % 8) * 8) * 4;

//             uint32_t bit_map_offset = x + y * image->extent.width;

//             float data[4];
//             data[0] = pixel_color[0] / 255.0;
//             data[1] = pixel_color[1] / 255.0;
//             data[2] = pixel_color[2] / 255.0;
//             data[3] = pixel_color[3] / 255.0;
//             imageFile.write((char*) data, 3 * sizeof(float));
//             imageFile.write((char*) (&bit_map_offset), sizeof(uint32_t));
//             imageFile.flush();
//         }
//     }
// }

// void save_ASTC_texture_to_text_file(const struct anv_image *image)
// {
//     FILE * pFile;
//     pFile = fopen ("ASTC_texture.txt","w");
//     assert(image->vk_format == VK_FORMAT_ASTC_8x8_SRGB_BLOCK);
//     uint8_t* address = anv_address_map(image->planes[0].address);

//     for(int block = 0; block * (128 / 8) < image->planes[0].size; block++)
//     {
//         uint8_t dst_colors[1024];
//         basisu::astc::decompress(dst_colors, address + block * (128 / 8), true, 8, 8);

//         fprintf(pFile, "block %d:", block);
//         for(int i = 0; i < 8; i++)
//         {
//             for(int j = 0; j < 8; j++)
//             {
//                 uint8_t* pixel_color = &dst_colors[0] + (i * 8 + j) * 4;
//                 fprintf(pFile, "\tpixel (%d, %d) = (%d, %d, %d, %d) = (%f, %f, %f, %f)\n", i, j, 
//                         pixel_color[0], pixel_color[1], pixel_color[2], pixel_color[3],
//                         pixel_color[0] / 255.0, pixel_color[1] / 255.0, pixel_color[2] / 255.0, pixel_color[3] / 255.0);
//             }
//             fprintf(pFile, "\n");
//         }
//         fprintf(pFile, "\n\n");
//     }

//     fclose (pFile);
// }

// void show_decompress_ASTC_texture_block(const uint8_t * data, bool isSRGB, int blockWidth, int blockHeight)
// {
//     uint8_t dst_colors[1024];
//     basisu::astc::decompress(dst_colors, data, isSRGB, blockWidth, blockHeight);

//     for(int i = 0; i < blockHeight; i++)
//     {
//         for(int j = 0; j < blockWidth; j++)
//         {
//             uint8_t* pixel_color = &dst_colors[0] + (i * blockWidth + j) * 4;
//             printf("pixel (%d, %d) = (%d, %d, %d, %d) = (%f, %f, %f, %f)\n", i, j, 
//                     pixel_color[0], pixel_color[1], pixel_color[2], pixel_color[3],
//                     pixel_color[0] / 255.0, pixel_color[1] / 255.0, pixel_color[2] / 255.0, pixel_color[3] / 255.0);
//         }
//         printf("\n");
//     }
// }


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

// uint32_t get_ASTC_block_offset(const struct anv_image *image, uint32_t x, uint32_t y);

#endif /* INTEL_IMAGE_H */