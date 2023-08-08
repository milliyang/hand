#include "stdafx.h"
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
    return zmq_connect(s_zmq.subscriber, addr);
}

static int imvt_zmq_sub_connect(char *addr)
{
    int rc;
#if 0
    //Leo FUCK:
    rc = _zmq_sub_connect(PUB_ADDR);
    if (rc == 0) {
        return OK;
    } else {
        return NG;
    }
#endif
    //char *filter = "10001 ";
    char *filter = ""; //subscrib all msg. no filter
    rc = zmq_setsockopt(s_zmq.subscriber, ZMQ_SUBSCRIBE, filter, strlen(filter));
    //rc = zmq_setsockopt(s_zmq.subscriber, ZMQ_SUBSCRIBE, NULL, 0);
    if (rc == 0) {
        return OK;
    } else {
        return NG;
    }
}

int imvt_zmq_sub_init(char *addr)
{
    int ret;
    memset(&s_zmq, 0, sizeof(s_zmq));
    s_zmq.context = zmq_ctx_new();
    s_zmq.subscriber = zmq_socket(s_zmq.context, ZMQ_SUB);
    //Leo FUCK:
    // must call connect immediatelly
    ret = _zmq_sub_connect(addr);
    if (ret == 0) {
        ret = imvt_zmq_sub_connect(NULL);
    }
    return ret;
}

int imvt_zmq_sub_recv_frame(unsigned char *p_buffer, int max_len)
{
    int size = zmq_recv (s_zmq.subscriber, p_buffer, max_len, 0);
    if (size == -1) {
        return NG;
    }
    return size;
}

int imvt_zmq_sub_deinit(void)
{
    zmq_close(s_zmq.subscriber);
    zmq_ctx_destroy(s_zmq.context);
    return OK;
}
