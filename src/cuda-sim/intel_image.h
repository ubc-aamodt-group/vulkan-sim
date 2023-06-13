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

#ifndef INTEL_IMAGE_H
#define INTEL_IMAGE_H

#include <fstream>
#include <vector>
#include <iostream>
#include <assert.h>
#include "astc_decomp.h"
#include "../abstract_hardware_model.h"

#include "anv_include.h"

#include "vulkan_ray_tracing.h"


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

Pixel load_image_pixel(const struct anv_image *image, uint32_t x, uint32_t y, uint32_t level, ImageMemoryTransactionRecord& transaction, uint64_t launcher_offset = 0)
{
    uint8_t *address;
    uint8_t *deviceAddress;
    uint32_t setID;
    uint32_t descID;
    uint64_t size;
    uint32_t width;
    uint32_t height;
    VkFormat vk_format;
    uint32_t VkDescriptorTypeNum;
    uint32_t n_planes;
    uint32_t samples;
    VkImageTiling tiling;
    isl_tiling isl_tiling_mode;
    uint32_t row_pitch_B;
    
    if (use_external_launcher)
    {
        texture_metadata *texture = (texture_metadata*) image;
        setID = texture->setID;
        descID = texture->descID;
        size = texture->size;
        width = texture->width;
        height = texture->height;
        vk_format = texture->format;
        VkDescriptorTypeNum = texture->VkDescriptorTypeNum;
        n_planes = texture->n_planes;
        samples = texture->n_samples;
        tiling = texture->tiling;
        isl_tiling_mode = texture->isl_tiling_mode;
        row_pitch_B = texture->row_pitch_B;
        address = (uint8_t*) texture->address;
        deviceAddress = (uint8_t*) texture->deviceAddress;
    }
    else
    {
        width = image->extent.width;
        height = image->extent.height;
        vk_format = image->vk_format;
        n_planes = image->n_planes;
        samples = image->samples;
        tiling = image->tiling;
        isl_tiling_mode = image->planes[0].surface.isl.tiling;
        row_pitch_B = image->planes[0].surface.isl.row_pitch_B;
        address = anv_address_map(image->planes[0].address);
    }

    assert(n_planes == 1);
    assert(samples == 1);
    assert(tiling == VK_IMAGE_TILING_OPTIMAL);
    assert(isl_tiling_mode == ISL_TILING_Y0);
    assert(level == 0);

    assert(0 <= x && x < width);
    assert(0 <= y && y < height);

    //uint8_t* address = anv_address_map(image->planes[0].address);

    switch(vk_format)
    {
        case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:
        {
            uint32_t tileWidth = 8;
            uint32_t tileHeight = 32;
            uint32_t ASTC_block_size = 128 / 8;

            int tileX = x / 8 / tileWidth;
            int tileY = y / 8 / tileHeight;
            int tileID = tileX + tileY * width / 8 / tileWidth;

            int blockX = ((x / 8) % tileWidth);
            int blockY = ((y / 8) % tileHeight);
            int blockID = blockX * (tileHeight) + blockY;

            uint32_t offset = (tileID * (tileWidth * tileHeight) + blockID) * ASTC_block_size;
            
            if (use_external_launcher)
            {
                transaction.address = deviceAddress + offset;
                transaction.size = 128 / 8;
            }
            else
            {
                transaction.address = address + offset;
                transaction.size = 128 / 8;
            }

            uint8_t dst_colors[256];
            if(!basisu::astc::decompress(dst_colors, address + offset, true, 8, 8))
            {
                printf("decoding error at pixel (%d, %d)\n", x, y);
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
            int tileID = tileX + tileY * width / tileWidth;

            if (use_external_launcher)
            {
                transaction.address = deviceAddress + (tileID * tileWidth * tileHeight + (x % tileWidth) * tileHeight + (y % tileHeight)) * 4;
                transaction.size = 4;
            }
            else
            {
                transaction.address = address + (tileID * tileWidth * tileHeight + (x % tileWidth) * tileHeight + (y % tileHeight)) * 4;
                transaction.size = 4;
            }

            uint8_t colors[4];

            intel_tiled_to_linear(x * 4, x * 4 + 4, y, y + 1,
                colors, address, width * 4 ,row_pitch_B, false,
                ISL_TILING_Y0, ISL_MEMCPY);

            Pixel pixel;
            pixel.r = SRGB_to_linearRGB(colors[0] / 255.0);
            pixel.g = SRGB_to_linearRGB(colors[1] / 255.0);
            pixel.b = SRGB_to_linearRGB(colors[2] / 255.0);
            pixel.a = colors[3] / 255.0;
            return pixel;
        }
        case VK_FORMAT_R8G8B8A8_UNORM:
        {
            uint32_t tileWidth = 32;
            uint32_t tileHeight = 32;
            int tileX = x / tileWidth;
            int tileY = y / tileHeight;
            int tileID = tileX + tileY * width / tileWidth;

            if (use_external_launcher)
            {
                transaction.address = deviceAddress + (tileID * tileWidth * tileHeight + (x % tileWidth) * tileHeight + (y % tileHeight)) * 4;
                transaction.size = 4;
            }
            else
            {
                transaction.address = address + (tileID * tileWidth * tileHeight + (x % tileWidth) * tileHeight + (y % tileHeight)) * 4;
                transaction.size = 4;
            }

            uint8_t colors[4];

            intel_tiled_to_linear(x * 4, x * 4 + 4, y, y + 1,
                colors, address, width * 4 ,row_pitch_B, false,
                ISL_TILING_Y0, ISL_MEMCPY);

            Pixel pixel;
            pixel.r = colors[0] / 255.0;
            pixel.g = colors[1] / 255.0;
            pixel.b = colors[2] / 255.0;
            pixel.a = colors[3] / 255.0;
            return pixel;
        }
        default:
        {
            printf("%d not implemented\n", vk_format);
            assert(0);
            break;
        }
    }
}

Pixel get_interpolated_pixel(struct anv_image_view *image_view, struct anv_sampler *sampler, float x, float y, std::vector<ImageMemoryTransactionRecord>& transactions, uint64_t launcher_offset = 0)
{
    uint32_t width;
    uint32_t height;
    VkFilter filter;

    const struct anv_image *image;
    
    if (use_external_launcher)
    {
        texture_metadata *texture = (texture_metadata*) image_view;
        width = texture->width;
        height = texture->height;
        filter = texture->filter;
        if(filter == NULL)
            filter = VK_FILTER_NEAREST;

        image = (anv_image*) image_view; // just for passing on the texture metadata
    }
    else
    {
        image = image_view->image;
        assert(sampler->conversion == NULL);

        if(sampler->conversion == NULL)
            filter = VK_FILTER_NEAREST;

        width = image->extent.width;
        height = image->extent.height;
    }
    
    
    if(x < 0 || x > 1)
        x -= std::floor(x);
    if(y < 0 || y > 1)
        y -= std::floor(y);
    
    
    switch (filter)
    {
        case VK_FILTER_NEAREST:
        {
            // uint32_t x_int = std::lround(x * image->extent.width);
            // uint32_t y_int = std::lround(y * image->extent.height);
            uint32_t x_int = std::floor(x * width);
            uint32_t y_int = std::floor(y * height);
            if(x_int >= width)
                x_int -= width;
            if(y_int >= height)
                y_int -= height;

            assert(0 <= x_int && x_int < width);
            assert(0 <= y_int && y_int < height);

            ImageMemoryTransactionRecord transaction;
            Pixel pixel = load_image_pixel(image, x_int, y_int, 0, transaction, launcher_offset);
            transactions.push_back(transaction);
            TXL_DPRINTF("Adding (nearest) txl transaction: 0x%x\n", transaction.address);
            return pixel;
        }
        case VK_FILTER_LINEAR:
        {
            // uint32_t xs[2];
            // xs[0] = std::floor(x * image->extent.width);
            // xs[1] = std::ceil(x * image->extent.width);
            
            // uint32_t ys[2];
            // ys[0] = std::floor(y * image->extent.height);
            // ys[1] = std::ceil(y * image->extent.height);

            int32_t xs[2];
            xs[0] = std::floor(x * width - 0.5);
            xs[1] = std::ceil(x * width - 0.5);
            
            int32_t ys[2];
            ys[0] = std::floor(y * height - 0.5);
            ys[1] = std::ceil(y * height - 0.5);
            
            Pixel pixel[2][2];
            float weight[2][2];

            for(int i = 0; i < 2; i++)
                for(int j = 0; j < 2; j++)
                {
                    // weight[i][j] = std::abs(x * image->extent.width - xs[(i + 1) % 2]) * std::abs(y * image->extent.height - ys[(j + 1) % 2]);
                    weight[i][j] = std::abs(x * width - (xs[(i + 1) % 2] + 0.5)) * std::abs(y * height - (ys[(j + 1) % 2] + 0.5));

                    int32_t xc = xs[i];
                    int32_t yc = ys[j];
                    if(xc >= (int)width)
                        xc -= width;
                    if(xc < 0)
                        xc += width;
                    if(yc >= (int)height)
                        yc -= height;
                    if(yc < 0)
                        yc += height;
                    
                    ImageMemoryTransactionRecord transaction;
                    pixel[i][j] = load_image_pixel(image, xc, yc, 0, transaction);

                    for(int i = 0; i < transactions.size(); i++)
                        if(transactions[i].address == transaction.address && transactions[i].size == transaction.size)
                            continue;
                    transactions.push_back(transaction);
                    TXL_DPRINTF("Adding (linear) txl transaction: 0x%x\n", transaction.address);
                }
            
            Pixel final_pixel(0, 0, 0, 0);
            for(int i = 0; i < 2; i++)
                for(int j = 0; j < 2; j++)
                {
                    final_pixel.c0 += pixel[i][j].c0 * weight[i][j];
                    final_pixel.c1 += pixel[i][j].c1 * weight[i][j];
                    final_pixel.c2 += pixel[i][j].c2 * weight[i][j];
                    final_pixel.c3 += pixel[i][j].c3 * weight[i][j];
                }
            return final_pixel;
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
    void *address;
    void *deviceAddress;
    uint32_t setID;
    uint32_t descID;
    uint32_t width;
    uint32_t height;
    VkFormat vk_format;
    uint32_t VkDescriptorTypeNum;
    uint32_t n_planes;
    uint32_t samples;
    VkImageTiling tiling;
    isl_tiling isl_tiling_mode; 
    uint32_t row_pitch_B;

    if (use_external_launcher)
    {
        storage_image_metadata *metadata = (storage_image_metadata*) image;
        setID = metadata->setID;
        descID = metadata->descID;
        width = metadata->width;
        height = metadata->height;
        vk_format = metadata->format;
        VkDescriptorTypeNum = metadata->VkDescriptorTypeNum;
        n_planes = metadata->n_planes;
        samples = metadata->n_samples;
        tiling = metadata->tiling;
        isl_tiling_mode = metadata->isl_tiling_mode;
        row_pitch_B = metadata->row_pitch_B;
        address = metadata->address;
        deviceAddress = metadata->deviceAddress;
    }
    else
    {
        width = image->extent.width;
        height = image->extent.height;
        vk_format = image->vk_format;
        n_planes = image->n_planes;
        samples = image->samples;
        tiling = image->tiling;
        isl_tiling_mode = image->planes[0].surface.isl.tiling;
        row_pitch_B = image->planes[0].surface.isl.row_pitch_B;
        address = anv_address_map(image->planes[0].address);
    }
    
    assert(n_planes == 1);
    assert(samples == 1);
    assert(level == 0);

    //void* address = anv_address_map(image->planes[0].address);

    switch (vk_format)
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

            switch (isl_tiling_mode)
            {
                case ISL_TILING_Y0:
                {
                    // uint32_t tileWidth = 32;
                    // uint32_t tileHeight = 32;
                    // int tileX = x / tileWidth;
                    // int tileY = y / tileHeight;
                    // int tileID = tileX + tileY * ceil_divide(image->extent.width, tileWidth);

                    // uint32_t offset = (tileID * tileWidth * tileHeight + (x % tileWidth) + (y % tileHeight) * tileWidth) * 4;
                    uint32_t ytile_span = 16;
                    uint32_t bytes_per_column = 512;
                    uint32_t ytile_height = 32;

                    uint32_t offset = (y / ytile_height) * ytile_height * row_pitch_B;
                    offset += (x * 4 % ytile_span) + (x * 4 / ytile_span) * bytes_per_column + (y % ytile_height) * ytile_span;

                    if (use_external_launcher)
                    {
                        transaction.address = deviceAddress + offset;
                        transaction.size = 4;
                    }
                    else
                    {
                        transaction.address = address + offset;
                        transaction.size = 4;
                    }

                    assert(tiling == VK_IMAGE_TILING_OPTIMAL);
                    intel_linear_to_tiled(x * 4, x * 4 + 4, y, y + 1,
                        (char *)address, colors, row_pitch_B, width * 4, false,
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

            switch (isl_tiling_mode)
            {
                case ISL_TILING_Y0:
                {
                    // uint32_t tileWidth = 32;
                    // uint32_t tileHeight = 32;
                    // int tileX = x / tileWidth;
                    // int tileY = y / tileHeight;
                    // int tileID = tileX + tileY * ceil_divide(image->extent.width, tileWidth);

                    // uint32_t offset = (tileID * tileWidth * tileHeight + (x % tileWidth) + (y % tileHeight) * tileWidth) * 4;

                    uint32_t ytile_span = 16;
                    uint32_t bytes_per_column = 512;
                    uint32_t ytile_height = 32;

                    uint32_t offset = (y / ytile_height) * ytile_height * row_pitch_B;
                    offset += (x * 4 % ytile_span) + (x * 4 / ytile_span) * bytes_per_column + (y % ytile_height) * ytile_span;

                    if (use_external_launcher)
                    {
                        transaction.address = deviceAddress + offset;
                        transaction.size = 4;
                    }
                    else
                    {
                        transaction.address = address + offset;
                        transaction.size = 4;
                    }

                    assert(tiling == VK_IMAGE_TILING_OPTIMAL);
                    intel_linear_to_tiled(x * 4, x * 4 + 4, y, y + 1,
                        (char *)address, colors, row_pitch_B, width * 4, false,
                        ISL_TILING_Y0, ISL_MEMCPY_BGRA8);
                    break;
                }

                case ISL_TILING_LINEAR:
                {
                    uint32_t offset = (y * width + x) * 4;

                    if (use_external_launcher)
                    {
                        transaction.address = deviceAddress + offset;
                        transaction.size = 4;
                    }
                    else
                    {
                        transaction.address = address + offset;
                        transaction.size = 4;
                    }

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

        case VK_FORMAT_R32G32B32A32_SFLOAT:
        {
            float colors[] = {pixel.r, pixel.g, pixel.b, pixel.a};

            switch (isl_tiling_mode)
            {
                case ISL_TILING_Y0:
                {
                    //MRS_TODO: check if transaction.address is calculated correctly
                    uint32_t ytile_span = 16;
                    uint32_t bytes_per_column = 512;
                    uint32_t ytile_height = 32;

                    uint32_t offset = (y / ytile_height) * ytile_height * row_pitch_B;
                    offset += (x * 16 % ytile_span) + (x * 16 / ytile_span) * bytes_per_column + (y % ytile_height) * ytile_span;

                    if (use_external_launcher)
                    {
                        transaction.address = deviceAddress + offset;
                        transaction.size = 4;
                    }
                    else
                    {
                        transaction.address = address + offset;
                        transaction.size = 4;
                    }

                    assert(tiling == VK_IMAGE_TILING_OPTIMAL);
                    intel_linear_to_tiled(x * 16, x * 16 + 16, y, y + 1,
                        (char *)address, (char*)colors, row_pitch_B, width * 4, false,
                        ISL_TILING_Y0, ISL_MEMCPY);
                    break;
                }
            
                default:
                {
                    assert(0);
                    break;
                }
            }
            break;
        }
        
        default:
        {
            assert(0);
            break;
        }
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
