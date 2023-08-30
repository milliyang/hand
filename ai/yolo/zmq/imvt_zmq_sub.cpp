//#include "stdafx.h"
#include "imvt_zmq_sub.h"
#include "zmq.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

struct imvt_mq_ctx {
    void *context;
    void *subscriber;
};
static struct imvt_mq_ctx s_zmq;

static int _zmq_sub_connect(char *addr)
{
    return -1;
}

static int imvt_zmq_sub_connect(char *addr)
{
    return -1;
}

int imvt_zmq_sub_init(char *addr)
{
    return -1;
}

int imvt_zmq_sub_recv_frame(unsigned char *p_buffer, int max_len)
{
    return -1;
}

int imvt_zmq_sub_deinit(void)
{
    return -1;
}
