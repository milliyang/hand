#pragma once

#ifndef __IVC_BASE_LOG_H__
#define __IVC_BASE_LOG_H__

#ifdef _WIN32
#define NONE
#define RED
#define LIGHT_RED
#define GREEN
#define LIGHT_GREEN
#define BLUE
#define LIGHT_BLUE
#define DARY_GRAY
#define CYAN
#define LIGHT_CYAN
#define PURPLE
#define LIGHT_PURPLE
#define BROWN
#define YELLOW
#define LIGHT_GRAY
#define WHITE
#else
#define NONE "\033[m"
#define RED "\033[0;32;31m"
#define LIGHT_RED "\033[1;31m"
#define GREEN "\033[0;32;32m"
#define LIGHT_GREEN "\033[1;32m"
#define BLUE "\033[0;32;34m"
#define LIGHT_BLUE "\033[1;34m"
#define DARY_GRAY "\033[1;30m"
#define CYAN "\033[0;36m"
#define LIGHT_CYAN "\033[1;36m"
#define PURPLE "\033[0;35m"
#define LIGHT_PURPLE "\033[1;35m"
#define BROWN "\033[0;33m"
#define YELLOW "\033[1;33m"
#define LIGHT_GRAY "\033[0;37m"
#define WHITE "\033[1;37m"
#endif

#ifndef LOG_TAG
#define LOG_TAG "null"
#endif

#define LOG_LEVEL_E	1
#define LOG_LEVEL_W 2
#define LOG_LEVEL_I 3
#define LOG_LEVEL_D 4
#define LOG_LEVEL_V 5

#ifdef LOG_IMPL
int imvt_log_init(const char *path);
int imvt_log(int level, const char * fmt, ...);

#define LOG_INIT imvt_log_init
#define LOGV(fmt, ...) imvt_log(LOG_LEVEL_V, "[%s] "fmt, LOG_TAG, ##__VA_ARGS__)
#define LOGI(fmt, ...) imvt_log(LOG_LEVEL_I, "[%s] "fmt, LOG_TAG, ##__VA_ARGS__)
#define LOGD(fmt, ...) imvt_log(LOG_LEVEL_D, "[%s] "fmt, LOG_TAG, ##__VA_ARGS__)
#define LOGW(fmt, ...) imvt_log(LOG_LEVEL_W, "[%s] "fmt, LOG_TAG, ##__VA_ARGS__)
#define LOGE(fmt, ...) imvt_log(LOG_LEVEL_E, "[%s] "fmt, LOG_TAG, ##__VA_ARGS__)
#else
#include <stdio.h>
#define LOG_INIT(a)
//c++11 fix: "[%s]\t" fmt"\n" space before fmt
#define LOGV(fmt, ...) fprintf(stderr, BLUE"[%s]\t" fmt, LOG_TAG, ##__VA_ARGS__)
#define LOGI(fmt, ...) fprintf(stderr, LIGHT_GREEN"[%s]\t" fmt, LOG_TAG, ##__VA_ARGS__)
#define LOGD(fmt, ...) fprintf(stderr, NONE"[%s]\t" fmt, LOG_TAG, ##__VA_ARGS__)
#define LOGW(fmt, ...) fprintf(stderr, BROWN"[%s]\t" fmt, LOG_TAG, ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, LIGHT_RED"[%s]\t" fmt, LOG_TAG, ##__VA_ARGS__)
// #define LOGV(fmt, ...) fprintf(stderr, BLUE"[%s]\t"fmt, LOG_TAG, ##__VA_ARGS__)
// #define LOGI(fmt, ...) fprintf(stderr, LIGHT_GREEN"[%s]\t"fmt, LOG_TAG, ##__VA_ARGS__)
// #define LOGD(fmt, ...) fprintf(stderr, NONE"[%s]\t"fmt, LOG_TAG, ##__VA_ARGS__)
// #define LOGW(fmt, ...) fprintf(stderr, BROWN"[%s]\t"fmt, LOG_TAG, ##__VA_ARGS__)
// #define LOGE(fmt, ...) fprintf(stderr, LIGHT_RED"[%s]\t"fmt, LOG_TAG, ##__VA_ARGS__)
#endif

#define log_module_v(module, fmt, ...) fprintf(stderr, BLUE"[%s]"fmt, module, ##__VA_ARGS__)
#define log_module_i(module, fmt, ...) fprintf(stderr, LIGHT_GREEN"[%s]"fmt, module, ##__VA_ARGS__)
#define log_module_d(module, fmt, ...) fprintf(stderr, NONE"[%s]"fmt, module, ##__VA_ARGS__)
#define log_module_w(module, fmt, ...) fprintf(stderr, BROWN"[%s]"fmt, module, ##__VA_ARGS__)
#define log_module_e(module, fmt, ...) fprintf(stderr, LIGHT_RED"[%s]"fmt, module, ##__VA_ARGS__)

#endif
