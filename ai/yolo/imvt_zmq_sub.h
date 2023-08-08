#ifndef __IMVT_ZMQ_SUB_H__
#define __IMVT_ZMQ_SUB_H__

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

#ifndef OK
#define OK (0)
#define NG (-1)
#endif

#define PUB_ADDR   "tcp://192.168.9.6:5556"

int imvt_zmq_sub_init(char *addr);
int imvt_zmq_sub_recv_frame(unsigned char *p_buffer, int max_len);
int imvt_zmq_sub_deinit(void);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */
#endif /* __IMVT_ZMQ_SUB_H__ */
