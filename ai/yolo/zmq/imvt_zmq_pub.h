#ifndef __IMVT_ZMQ_PUB_H__
#define __IMVT_ZMQ_PUB_H__

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

#ifndef OK
#define OK (0)
#define NG (-1)
#endif

int imvt_zmq_pub_init(void);
int imvt_zmq_pub_bind(void);
int imvt_zmq_pub_send_frame(unsigned char *p_buffer, int len);
int imvt_zmq_pub_send_frame_x(unsigned char *p_buffer, int len, unsigned char *p_buffer2, int len2);
int imvt_zmq_pub_deinit(void);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */
#endif /* __IMVT_ZMQ_PUB_H__ */
