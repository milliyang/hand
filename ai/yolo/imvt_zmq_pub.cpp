#include "stdafx.h"
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
    memset(&s_zmq, 0, sizeof(s_zmq));
    s_zmq.context = zmq_ctx_new();
    s_zmq.publisher = zmq_socket(s_zmq.context, ZMQ_PUB);

    int hwm = HIGH_WATER_MARK;
    // set high water mark. (max queue for outgoing)
    int rc = zmq_setsockopt (s_zmq.publisher, ZMQ_SNDHWM, &hwm, sizeof(int));
    if (rc == 0) {
        return OK;
    } else {
        return NG;
    }
}
int imvt_zmq_pub_bind(void)
{
    int rc = zmq_bind(s_zmq.publisher, PUB_ADDR);
    if (rc == 0) {
        return OK;
    } else {
        return NG;
    }
}

int imvt_zmq_pub_send_frame_x(unsigned char *p_buffer, int len, unsigned char *p_buffer2, int len2)
{
    unsigned char * p;
    zmq_msg_t message;
    zmq_msg_init_size (&message, len + len2);
    p = (unsigned char *)zmq_msg_data (&message);
    memcpy (p, p_buffer, len);
    p+=len;
    memcpy (p, p_buffer2, len2);

    int size = zmq_msg_send (&message, s_zmq.publisher, 0);
    zmq_msg_close (&message);
    return (size);
}

int imvt_zmq_pub_send_frame(unsigned char *p_buffer, int len)
{
    zmq_msg_t message;
    zmq_msg_init_size (&message, len);
    memcpy (zmq_msg_data (&message), p_buffer, len);
    int size = zmq_msg_send (&message, s_zmq.publisher, 0);
    zmq_msg_close (&message);
    return (size);
}

int imvt_zmq_pub_deinit(void)
{
    zmq_close(s_zmq.publisher);
    zmq_ctx_destroy(s_zmq.context);
    return OK;
}
