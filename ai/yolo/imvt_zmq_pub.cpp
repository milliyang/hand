//#include "stdafx.h"

#include "imvt_zmq_pub.h"
#include "zmq.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// sometime no working, must use specify addr
#define PUB_ADDR "tcp://*:5556"
// #define PUB_ADDR "tcp://192.168.9.6:5556"
#define HIGH_WATER_MARK   (10)

struct imvt_mq_ctx {
    void *context;
    void *publisher;
};
static struct imvt_mq_ctx s_zmq;

int imvt_zmq_pub_init(void)
{
return OK;
}
int imvt_zmq_pub_bind(void)
{
return OK;
}

int imvt_zmq_pub_send_frame_x(unsigned char *p_buffer, int len, unsigned char *p_buffer2, int len2)
{
return OK;
}

int imvt_zmq_pub_send_frame(unsigned char *p_buffer, int len)
{
return OK;
}

int imvt_zmq_pub_deinit(void)
{
    return OK;
}
